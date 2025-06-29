import json
from itertools import chain, product
from pathlib import Path

import librosa
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.tasks.s2t import S2TTask

from utils import preprocessor_add_new_tokens


task_id = {
    "transcription": "asr",
    "gloss": "st_gloss",
    "underlying": "st_underlying",
    "translation": "st_eng",
}

class FieldworkDataset(Dataset):
    def __init__(self, root, split, tasks, langs, preprocessor):
        assert split in ("train", "dev", "test")
        assert all((t in ("transcription", "underlying", "gloss", "translation")) for t in tasks)

        self.split = split
        self.data = self._parse_data(root, tasks, langs)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index].copy()
        
        audio_path = row["speech"]
        row["speech"], _ = librosa.load(audio_path, mono=True, sr=16_000)
        audio_filename = Path(audio_path).name
        
        uid = f"{index:08d}_{audio_filename}"

        return uid, self.preprocessor(uid=uid, data=row)

    def _parse_data(self, root, tasks, langs):
        paths = [root / f"data/{lang}/{self.split}.json" for lang in langs]
        self.tasks = tasks
        self.langs = [lang for path, lang in zip(paths, langs) if path.exists()]
        return list(chain.from_iterable(
            self._parse_single_data(path, task)
            for path, task in product(paths, tasks)
            if path.exists()
        ))

    def _parse_single_data(self, path, task):
        with open(path) as f:
            meta = json.load(f)

        data = []
        for key, value in meta.items():
            if value["discard"]:
                continue
            if len(value[task]) == 0:
                continue
            if task == "translation" and value["translation_language"] != "en":
                continue

            speech = path.parent / "audio" / path.stem / key
            text_ctc = value["transcription"]
            text_prev = "<na>"

            lang = path.parent.name
            text = f"<{lang}><{task_id[task]}>{value[task]}"

            if (self.split == "test") or (
                (librosa.get_duration(filename=speech) <= 30.0)
            ):
                data.append({
                    "speech": speech,
                    "text": text,
                    "text_prev": text_prev,
                    "text_ctc": text_ctc,
                })
        return data


class FieldworkDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_path: Path,
            model_name: str,
            batch_size: int,
            new_tokens: list[str],
            tasks: list[str],
            langs: list[str],
            num_workers: int,
            **kwargs,
        ):
        super().__init__()

        s2t = Speech2Text.from_pretrained(model_name)

        preprocessor_train = S2TTask.build_preprocess_fn(s2t.s2t_train_args, train=True)
        preprocessor_test = S2TTask.build_preprocess_fn(s2t.s2t_train_args, train=False)
        preprocessor_add_new_tokens(preprocessor_train, new_tokens)
        preprocessor_add_new_tokens(preprocessor_test, new_tokens)

        self.collator_train = S2TTask.build_collate_fn(s2t.s2t_train_args, train=True)
        self.collator_test = S2TTask.build_collate_fn(s2t.s2t_train_args, train=False)

        self.unk_id = preprocessor_train.token_id_converter.tokens2ids(["<unk>"])[0]

        self.train_ds = FieldworkDataset(
            dataset_path, "train", tasks, langs, preprocessor_train,
        )

        self.valid_ds = {
            f"{task}_{lang}": FieldworkDataset(
                dataset_path, "dev", [task], [lang], preprocessor_test)
            for task, lang in product(tasks, langs)
        }
        self.valid_ds = {k: v for k, v in self.valid_ds.items() if len(v) > 0}

        self.test_ds = {
            f"{task}_{lang}": FieldworkDataset(
                dataset_path, "test", [task], [lang], preprocessor_test)
            for task, lang in product(tasks, langs)
        }
        self.test_ds = {k: v for k, v in self.test_ds.items() if len(v) > 0}

        del s2t

        self.num_workers = num_workers
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator_train,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator_test,
                persistent_workers=True,
            )
            for ds in self.valid_ds.values()
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collator_test,
            )
            for ds in self.test_ds.values()
        ]
