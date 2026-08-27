# -*- coding: utf-8 -*-
"""将多个结果表分节写入单个CSV。"""
import pandas as pd


class CsvCollector:

    def __init__(self):
        self._blocks = []

    def add(self, section, df, index=False):
        self._blocks.append((section, df, bool(index)))
        return df

    def save(self, path, encoding='utf-8-sig'):
        with open(path, 'w', encoding=encoding, newline='') as f:
            for i, (section, df, index) in enumerate(self._blocks):
                if i:
                    f.write('\n')
                f.write(f'#  {section} \n')
                df.to_csv(f, index=index)
