from collections import defaultdict

import numpy as np
import pandas as pd

from utils.parsing import process_surgery_part, process_surgery, process_deviation, log_mar

# noinspection SpellCheckingInspection
TEST_CASES_FOR_PARTS = (
    # INPUT               # OUTPUT
    ('RE SR Recess 3 mm',      {'eyes': ('RE', ), 'muscle': 'SR', 'num_mm': (-3.0, )}),
    ('RE Myectomy of IO',      {'eyes': ('RE', ), 'muscle': 'IO', 'category': 'Myectomy'}),
    ('Advance MR 3 mm',        {'muscle': 'MR', 'num_mm': (3.0, )}),
    ('RE Rsect LR 6 mm',       {'eyes': ('RE', ), 'muscle': 'LR', 'num_mm': (6.0, )}),
    ('Supra',                  {'category': 'Supra'}),
    ('BE IO Myectomy',         {'eyes': ('LE', 'RE'), 'muscle': 'IO', 'category': 'Myectomy'}),
    ('IO Anteroposition',      {'muscle': 'IO', 'category': 'Anteroposition'}),
    ('BE IO 14 mm',            {'eyes': ('LE', 'RE'), 'muscle': 'IO', 'num_mm': (14.0, 14.0)}),
    ('Myectomy of adhesions of IO', {'muscle': 'IO', 'category': 'Myectomy of adhesions'}),
    ('Kestenbaum procedure',   {'category': 'Kestenbaum procedure'}),
    ('BE Recess IO L>>R',      {'eyes': ('LE', 'RE'), 'muscle': 'IO', 'category': 'L>>R'}),
    ('BE Recess MR 5.75/6.75', {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'num_mm': (-6.75, -5.75)}),
    ('BE Recess MR R 3.5 mm L 4 mm', {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'num_mm': (-4.0, -3.5)}),
    ('BE Recess MR 5.25',      {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'num_mm': (-5.25, -5.25)}),
    ('BE Recess MR 4.5 MM',    {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'num_mm': (-4.5, -4.5)}),
    ('BE Supraplacement of MR', {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'category': 'Supraplacement'}),
    ('BE Recess LR R 6mm with Y splitting L 7mm', {'eyes': ('LE', 'RE'), 'muscle': 'LR', 'num_mm': (-7.0, -6.0), 'category': 'Y splitting'}),
    ('BERecess MR 3mm',        {'eyes': ('LE', 'RE'), 'muscle': 'MR', 'num_mm': (-3.0, -3.0)}),
)


def test_parse_surgery_part() -> None:
    for part, expected_output in TEST_CASES_FOR_PARTS:
        actual = process_surgery_part(part)
        assert actual == expected_output, f"Input: {part}"


# noinspection SpellCheckingInspection
TEST_CASES_FOR_WHOLE = (
    # INPUT                                          # OUTPUT
    ('LE Recess LR 7.5 mm',                         {'LE': {'LR': {'num_mm': -7.5}}}),
    ('18/9/97 RE Myectomy of IO+RE SR Recess 3 mm', {'date': '1997-09-18 00:00', 'RE': {'IO': {'category': 'Myectomy'}, 'SR': {'num_mm': -3.0}}}),
    ('RE Recess LR 5 mm + Advance MR 3 mm',         {'RE': {'LR': {'num_mm': -5.0}, 'MR': {'num_mm': 3.0}}}),
    ('RE Recess MR 4 mm +RE Rsect LR 6 mm + Supra', {'RE': {'MR': {'num_mm': -4.0}, 'LR': {'num_mm': 6.0}}, 'unprocessed': 'Supra'}),
    ('BE Recess MR 4.75 mm+BE IO Myectomy',         {'LE': {'MR': {'num_mm': -4.75}, 'IO': {'category': 'Myectomy'}}, 'RE': {'MR': {'num_mm': -4.75}, 'IO': {'category': 'Myectomy'}}}),
    ('BE Recess MR 5.5 mm+IO Anteroposition',       {'LE': {'MR': {'num_mm': -5.5}, 'IO': {'category': 'Anteroposition'}}, 'RE': {'MR': {'num_mm': -5.5}, 'IO': {'category': 'Anteroposition'}}}),
    ('BE Recess IO 14 mm+ RE Advance MR 2 mm',      {'LE': {'IO': {'num_mm': -14.0}}, 'RE': {'IO': {'num_mm': -14.0}, 'MR': {'num_mm': 2.0}}}),
    ('RE Recess SR 4 mm+Resect MR 2.5 mm+ Myectomy of adhesions of IO',
                                                    {'RE': {'SR': {'num_mm': -4.0}, 'MR': {'num_mm': 2.5}, 'IO': {'category': 'Myectomy of adhesions'}}}),
    ('RE Recess LR 9.5 mm+advance MR 6 mm+LE Recess MR 4.5 mm+Resect LR 10 mm\r\nKestenbaum procedure',
                                                    {'RE': {'LR': {'num_mm': -9.5}, 'MR': {'num_mm': 6.0}}, 'LE': {'MR': {'num_mm': -4.5}, 'LR': {'num_mm': 10.0}}, 'Kestenbaum': 1}),
    ('BE Recess MR 5 mm',                           {'LE': {'MR': {'num_mm': -5.0}}, 'RE': {'MR': {'num_mm': -5.0}}}),
    ('Kestenbaum procediure\r\nRE Recess LR 6 mm+ Resect MR 6.75 mm\r\nLE Recess MR 6.5 mm+Resect LR 10.5 mm',
                                                    {'RE': {'LR': {'num_mm': -6.0}, 'MR': {'num_mm': 6.75}}, 'LE': {'MR': {'num_mm': -6.5}, 'LR': {'num_mm': 10.5}}, 'Kestenbaum': 1}),
    ('BE Recess MR 6.5 mm + IO Myectomy',           {'LE': {'MR': {'num_mm': -6.5}, 'IO': {'category': 'Myectomy'}}, 'RE': {'MR': {'num_mm': -6.5}, 'IO': {'category': 'Myectomy'}}}),
    ('BE Recess MR  diagonal 5.75/6.75+ BE MR supraplacement',
                                                    {'LE': {'MR': {'num_mm': -6.75, 'category': 'diagonal Supraplacement'}}, 'RE': {'MR': {'num_mm': -5.75, 'category': 'diagonal Supraplacement'}}}),
    ('BE Recess LR 5 mm + RE Y spliting of LR',     {'LE': {'LR': {'num_mm': -5.0}}, 'RE': {'LR': {'num_mm': -5.0, 'category': 'Y splitting'}}}),
    ('LE Recess LR 9.5 mm + Resect MR 3.5 mm +advance MR 3 mm',
                                                    {'LE': {'LR': {'num_mm': -9.5}, 'MR': {'num_mm': 6.5}}}),
    ('BE Recess MR R 3.5 mm L 4 mm+ IO Myectomy',  {'LE': {'MR': {'num_mm': -4.0}, 'IO': {'category': 'Myectomy'}}, 'RE': {'MR': {'num_mm': -3.5}, 'IO': {'category': 'Myectomy'}}}),
    ('15/10/98 BE Recess LR 6 mm\n11/11/99BE Resect MR 3.5 mm', {'unprocessed': '15/10/98 BE Recess LR 6 mm\n11/11/99BE Resect MR 3.5 mm'}),
    ('RE Recess LR 5 mm+ Resect MR 3 mm\nRE Synechiolysis of IO', {'RE': {'LR': {'num_mm': -5.0}, 'MR': {'num_mm': 3.0},  'IO': {'category': 'Synechiolysis'}}}),
    ('1987 LE Recess LR 7 mm+ Resect MR 5 mm + infraplace\n2001 LE Recess LR 5 mm from insertion on adjustable',
                                                    {'unprocessed': '1987 LE Recess LR 7 mm+ Resect MR 5 mm + infraplace\n2001 LE Recess LR 5 mm from insertion on adjustable'}),
    ('Kestenbaum procedure\r\nRE Recess MR 6.75 LR Resect 10.75\r\nLE Recess LR 10 MR Resect 8 mm',
                                                    {'RE': {'MR': {'num_mm': -6.75}, 'LR': {'num_mm': 10.75}}, 'LE': {'LR': {'num_mm': -10.0}, 'MR': {'num_mm': 8.0}}, 'Kestenbaum': 1}),
    ('LE Recess LR 3   Syenchiolysis+\r\nLE Resect MR 4 mm + Advance 3 mm',
                                                    {'LE': {'LR': {'num_mm': -3.0, 'category': 'Syenchiolysis'}, 'MR': {'num_mm': 7.0}}}),
    ('BE Recess Mr 2 mm + Post. fixation suture',   {'LE': {'MR': {'num_mm': -2.0}}, 'RE': {'MR': {'num_mm': -2.0}}, 'Post. Fixation suture': 1}),
    ('LE Recess 9 mm+Resect MR 6 mm 16/9/99',       {'date': '1999-09-16 00:00', 'LE': {'LR': {'num_mm': -9.0}, 'MR': {'num_mm': 6.0}}}),
    ('BE RecessMR 5mm + Recess IO 14mm',            {'LE': {'MR': {'num_mm': -5.0}, 'IO': {'num_mm': -14.0}}, 'RE': {'MR': {'num_mm': -5.0}, 'IO': {'num_mm': -14.0}}}),
    ('Re Recess LR 5 mm + Resect MR 3 mm',          {'RE': {'LR': {'num_mm': -5.0}, 'MR': {'num_mm': 3.0}}}),
    ('Kestenbaum procedure: RE recess MR 6 MM +Resect LR 9.5 mm\r\nLE Recess LR 8.5 mm +Resect MR 7.25 mm',
                                                    {'RE': {'MR': {'num_mm': -6.0}, 'LR': {'num_mm': 9.5}}, 'LE': {'LR': {'num_mm': -8.5}, 'MR': {'num_mm': 7.25}}, 'Kestenbaum': 1}),
    ('RE Recess 6mm + LR Resect 8.75mm',            {'RE': {'LR': {'num_mm': 8.75}, 'MR': {'num_mm': -6.0}}}),
)


def test_parse_surgery_whole() -> None:
    for part, expected_output in TEST_CASES_FOR_WHOLE:
        actual = process_surgery(part)
        assert actual == expected_output, f"Input: {part}"


# noinspection SpellCheckingInspection
TEST_CASES_FOR_DEVIATIONS = (
    # INPUT                                                              # OUTPUT
    ({'Dsc1Free': 30, 'Dsc1': 'XT', 'Dsc2Free': 25, 'Dsc2': 'RHT'},      {'Resolved_Dsc': 30}, tuple()),
    ({'Dcc1Free': np.nan, 'Dcc1': 'Ortho'},                              {'Resolved_Dcc': 0}, tuple()),
    ({'Nsc1FreeLine2': '0-30', 'Nsc1Line2': 'ET'},                       {}, ('0-30 ET', )),
    ({'Dcc1FreeLine2': np.nan, 'Dcc1Line2': '16x'},                      {'Resolved_Dcc': 16}, tuple()),
    ({'Dcc1FreeLine2': np.nan, 'Dcc1Line2': '+14'},                      {}, ('NAN +14', )),
    ({'Nsc1': 'flicke', 'Nsc1Free': np.nan},                             {'Resolved_Nsc': -2}, tuple()),
    ({'Dsc1Free': np.nan, 'Dsc1': 'Ortho', 'Dsc2Free': 'to', 'Dsc2': 'E(T)'}, {'Resolved_Dsc': 0}, ('TO E(T)', )),
    ({'Dcc1Free': 'Flick', 'Dcc1': 'ET'},                                {'Resolved_Dcc': -2}, tuple()),
    ({'Dsc1Free': '10 deg', 'Dsc1': 'RXT'},                              {'Resolved_Dsc': 10}, tuple()),
    ({'Dcc1FreeLine2': 0, 'Ncc1FreeLine2': 12, 'Ncc1Line2': 'E'},        {'Resolved_Dcc': 0, 'Resolved_Ncc': -12}, tuple()),
    ({'AddCCLine2': 2, 'AddCC1FreeLine2': 0},                            {'Resolved_AddCC': 0}, tuple()),
)


def test_process_deviation() -> None:
    for input_dict, expected_output, expected_ignored in TEST_CASES_FOR_DEVIATIONS:
        ignored = defaultdict(list)
        actual = process_deviation(pd.Series({'Code': 1, **input_dict}), ignored=ignored).to_dict()
        assert actual == expected_output, f"Input: {input_dict}"
        assert dict(ignored) == {i: [1] for i in expected_ignored}


TEST_CASES_FOR_LOG_MAR = (
    # INPUT      # OUTPUT
    ("6/6",      0),
    ("6/6-",     -0.02),
    ("C-S-M",    np.nan),
    ("6/7.5++",  -np.log10(6/7.5) + 0.04),
    ("6/9-+",    -np.log10(6/9)),
)


def test_log_mar() -> None:
    for input_s, expected_output in TEST_CASES_FOR_LOG_MAR:
        actual = log_mar(input_s)
        assert (actual == expected_output) or (np.isnan(actual) and np.isnan(expected_output)), f"Input: {input_s}"
