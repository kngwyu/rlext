""" Record class that stores {Key: List[Value]} dict
"""
import atexit
import typing as t
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd


class _StdoutConfig(t.NamedTuple):
    interval: int
    indices: t.List[str]
    color: str
    groupby: str


def _save_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open(mode="a") as f:
        df.to_json(f, orient="records", lines=True)


class Record:
    """Available colors: black, red, green, yellow, blue, magenta, cyan, white"""

    def __init__(
        self,
        *,
        stdout_interval: t.Optional[int] = None,
        stdout_indices: t.List[str] = [],
        stdout_color: str = "red",
        stdout_groupby: t.Optional[str] = None,
        save_path: t.Optional[Path] = None,
        save_interval: t.Optional[int] = None,
        save_fn: t.Callable[[pd.DataFrame, Path], None] = _save_jsonl,
    ) -> None:
        self._records = defaultdict(list)
        self._stdout_config = _StdoutConfig(
            stdout_interval,
            stdout_indices,
            stdout_color,
            stdout_groupby,
        )
        self._save_path = save_path
        self._save_interval = save_interval
        self._save_fn = save_fn
        atexit.register(self._dump)

    def submit(self, d: t.Dict[str, t.Any]) -> int:
        max_length = 0
        has_required_keys = False
        interval, indices, *_ = self._stdout_config
        for key, value in d.items():
            self._records[key].append(value)
            max_length = max(max_length, len(self._records[key]))
            has_required_keys |= key in indices
        if len(indices) > 0 and not has_required_keys:
            raise ValueError(
                f"Submitted log does not contain the required keys {indices}"
            )
        if interval is not None and max_length % interval == 0:
            self._summarize()
        if self._save_interval is not None and max_length % self._save_interval == 0:
            self._dump(truncate=True)
        return max_length

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def reset(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return "Record({})".format(repr(self._records))

    def _dump(self, truncate: bool = False) -> None:
        if self._save_path is None:
            return

        df = pd.DataFrame(self._records)
        self._save_fn(df, self._save_path)
        if truncate:
            interval = self._stdout_config[0]
            del_range = slice(None) if interval is None else slice(-interval)
            for records in self._records.values():
                records.clear(del_range)

    def _summarize(self) -> None:
        interval, indices, color, groupby = self._stdout_config
        recent = {}
        for key, value in self._records.items():
            recent[key] = value[-interval:]
        root_df = pd.DataFrame(recent)
        if groupby is None:
            groupby_iter = [(0, main_df)]
            groupby = ""
        else:
            groupby_iter = root_df.groupby(groupby)

        for value, df in groupby_iter:
            indices_df = df[indices]
            click.secho(
                f"============ {groupby}: {value} =============",
                bg=color,
                fg="white",
                bold=True,
            )
            min_, max_ = indices_df.iloc[0], indices_df.iloc[-1]
            range_str = "\n".join(
                [f"{idx}: {min_[idx]}-{max_[idx]}" for idx in indices]
            )
            click.secho(range_str, bg="black", fg="white")
            df.drop(columns=indices + [groupby], inplace=True)
            describe = df.describe()
            describe.drop(labels="count", inplace=True)
            click.echo(describe)
