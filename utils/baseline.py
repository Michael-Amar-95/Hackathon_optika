from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

MONO_BINO = 'mono/bino'
MONO = 'Monocular'
BINO = 'Binocular'


@dataclass
class PointSeries:
    surgery_mm: list[float]
    deviations: list[float]

    def __post_init___(self) -> None:
        assert len(self.surgery_mm) == len(self.deviations), "Lengths of surgery_mm and deviations must match"

    def __call__(self, surgery_mm: pd.Series) -> pd.Series:
        return pd.Series(index=surgery_mm.index, data=np.interp(surgery_mm, xp=self.surgery_mm, fp=self.deviations))


@dataclass
class Baseline:
    mr_mono_positive: PointSeries
    mr_mono_negative: PointSeries
    mr_bino_positive: PointSeries
    mr_bino_negative: PointSeries
    lr_mono_positive: PointSeries
    lr_mono_negative: PointSeries
    lr_bino_positive: PointSeries
    lr_bino_negative: PointSeries

    @staticmethod
    def dev_diff_baseline(positive_series: PointSeries, negative_series: PointSeries, num_mm: pd.Series) -> pd.Series:
        return positive_series(num_mm).where(num_mm > 0, (- negative_series(-num_mm)).where(num_mm < 0, 0))

    def dev_diff_baseline_for_mr_mono(self, num_mm: pd.Series) -> pd.Series:
        return - self.dev_diff_baseline(
            self.mr_mono_positive, self.mr_mono_negative, num_mm
        )

    def dev_diff_baseline_for_mr_bino(self, num_mm: pd.Series) -> pd.Series:
        return - self.dev_diff_baseline(
            self.mr_bino_positive, self.mr_bino_negative, num_mm
        )

    def dev_diff_baseline_for_lr_mono(self, num_mm: pd.Series) -> pd.Series:
        return self.dev_diff_baseline(
            self.lr_mono_positive, self.lr_mono_negative, num_mm
        )

    def dev_diff_baseline_for_lr_bino(self, num_mm: pd.Series) -> pd.Series:
        return self.dev_diff_baseline(
            self.lr_bino_positive, self.lr_bino_negative, num_mm
        )

    def generate_prediction(self, df: pd.DataFrame) -> pd.Series:
        assert df[MONO_BINO].isin([MONO, BINO]).all(), f"Column {MONO_BINO} must contain only {MONO} or {BINO}"
        return cast(pd.Series, (
            self.dev_diff_baseline_for_mr_mono(df['LE_MR_num_mm']) + self.dev_diff_baseline_for_mr_mono(df['RE_MR_num_mm'])
            + self.dev_diff_baseline_for_lr_mono(df['LE_LR_num_mm']) + self.dev_diff_baseline_for_lr_mono(df['RE_LR_num_mm'])
        ) / 2).where(
            df[MONO_BINO] == MONO,
            (
                self.dev_diff_baseline_for_mr_bino(df['LE_MR_num_mm']) + self.dev_diff_baseline_for_mr_bino(df['RE_MR_num_mm'])
                + self.dev_diff_baseline_for_lr_bino(df['LE_LR_num_mm']) + self.dev_diff_baseline_for_lr_bino(df['RE_LR_num_mm'])
            ) / 2,
        )


HAIM_BASELINE = Baseline(
    mr_mono_positive=PointSeries(surgery_mm=[3, 4, 5, 6], deviations=[15, 25, 35, 45]),
    mr_mono_negative=PointSeries(surgery_mm=[3, 4, 5, 6], deviations=[15, 25, 35, 45]),
    mr_bino_positive=PointSeries(surgery_mm=[3, 4, 5, 6], deviations=[15, 25, 35, 45]),
    mr_bino_negative=PointSeries(surgery_mm=[3, 4, 5, 6], deviations=[15, 25, 35, 45]),
    lr_mono_positive=PointSeries(surgery_mm=[5, 6, 7.5, 9], deviations=[15, 25, 35, 45]),
    lr_mono_negative=PointSeries(surgery_mm=[5, 6, 7.5, 9], deviations=[15, 25, 35, 45]),
    lr_bino_positive=PointSeries(surgery_mm=[5, 6, 7.5, 9], deviations=[15, 25, 35, 45]),
    lr_bino_negative=PointSeries(surgery_mm=[5, 6, 7.5, 9], deviations=[15, 25, 35, 45]),
)

WRITE_BASELINE = Baseline(
    mr_mono_positive=PointSeries(surgery_mm=[3, 4, 4.5, 5, 5.5, 6, 6.5], deviations=[15, 20, 25, 30, 35, 40, 50]),
    mr_mono_negative=PointSeries(surgery_mm=[3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], deviations=[15, 20, 25, 30, 35, 40, 50, 60, 70]),
    mr_bino_positive=PointSeries(surgery_mm=[3, 4, 5, 5.5, 6, 6.5], deviations=[15, 20, 25, 30, 35, 40]),
    mr_bino_negative=PointSeries(surgery_mm=[3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7], deviations=[15, 20, 25, 30, 35, 40, 50, 60, 70]),
    lr_mono_positive=PointSeries(surgery_mm=[3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8], deviations=[15, 20, 25, 30, 35, 40, 50, 60, 70]),
    lr_mono_negative=PointSeries(surgery_mm=[4, 5, 6, 6.5, 7, 7.5, 8.5], deviations=[15, 20, 25, 30, 35, 40, 50]),
    lr_bino_positive=PointSeries(surgery_mm=[3.5, 4.5, 5.5, 6, 6.5, 7, 8], deviations=[15, 20, 25, 30, 35, 40, 50]),
    lr_bino_negative=PointSeries(surgery_mm=[4, 5, 6, 7, 7.5, 8, 9], deviations=[15, 20, 25, 30, 35, 40, 50]),
)

PARKS_BASELINE = Baseline(
    mr_mono_positive=PointSeries(surgery_mm=[3, 4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 35, 50, 60, 70, 80]),
    mr_mono_negative=PointSeries(surgery_mm=[3, 3.5, 4, 4.5, 5, 5.5, 6, 7], deviations=[15, 20, 25, 30, 35, 40, 50, 60]),
    mr_bino_positive=PointSeries(surgery_mm=[3, 4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 35, 50, 60, 70, 80]),
    mr_bino_negative=PointSeries(surgery_mm=[3, 3.5, 4, 4.5, 5, 5.5, 6, 7], deviations=[15, 20, 25, 30, 35, 40, 50, 60]),
    lr_mono_positive=PointSeries(surgery_mm=[4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 30, 35, 40, 50]),
    lr_mono_negative=PointSeries(surgery_mm=[4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 30, 40, 50, 60]),
    lr_bino_positive=PointSeries(surgery_mm=[4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 30, 35, 40, 50]),
    lr_bino_negative=PointSeries(surgery_mm=[4, 5, 6, 7, 8, 9, 10], deviations=[15, 20, 25, 30, 40, 50, 60]),
)

BASELINES = {
    'Haim': HAIM_BASELINE,
    'Write': WRITE_BASELINE,
    'Parks': PARKS_BASELINE,
}
