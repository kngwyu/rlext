""" Records stores {Key: List[Value]} dict
"""
import typing as t
from collections import defaultdict

import click
import pandas as pd


class _StdoutConfig(t.NamedTuple):
    interval: int
    indices: t.List[str]
    color: str
    describe_range: bool


class Records:
    """Available colors: black, red, green, yellow, blue, magenta, cyan, white"""

    def __init__(
        self,
        name: str,
        *,
        stdout_interval: t.Optional[int] = None,
        stdout_indices: t.List[str] = [],
        stdout_color: str = "red",
        stdout_describe_range: bool = False,
    ) -> None:
        self.name = name
        self._records = defaultdict(list)
        self._stdout_config = _StdoutConfig(
            stdout_interval,
            stdout_indices,
            stdout_color,
            stdout_describe_range,
        )

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
        return max_length

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def reset(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return "Record({})".format(repr(self._records))

    def _summarize(self) -> None:
        interval, indices, color, describe_range = self._stdout_config
        recent = {}
        for key, value in self._records.items():
            recent[key] = value[-interval:]
        df = pd.DataFrame(recent)
        indices_df = df[indices]
        click.secho(
            f"============ {self.name.upper()} =============",
            bg=color,
            fg="white",
            bold=True,
        )
        min_, max_ = indices_df.iloc[0], indices_df.iloc[-1]
        range_str = "\n".join([f"{idx}: {min_[idx]}-{max_[idx]}" for idx in indices])
        click.secho(range_str, bg="black", fg="white")
        df.drop(columns=indices, inplace=True)
        print(describe_range)
        if describe_range:
            min_, max_ = df.iloc[0], df.iloc[-1]
            click.echo("\n".join([f"{i}: {min_[i]}-{max_[i]}" for i in df.columns]))
        else:
            describe = df.describe()
            describe.drop(labels="count", inplace=True)
            click.echo(describe)
