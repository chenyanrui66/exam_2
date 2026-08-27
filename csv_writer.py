# -*- coding: utf-8 -*-
"""将多个结果表分节写入单个CSV。"""
import io

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


def read_sections(path, encoding='utf-8-sig'):
    """读取 CsvCollector 生成的分节CSV，返回 {节名: DataFrame}。"""
    sections = {}
    name, buf = None, []
    with open(path, encoding=encoding) as f:
        for line in f:
            if line.startswith('# '):
                if name is not None and buf:
                    sections[name] = pd.read_csv(io.StringIO(''.join(buf)))
                name = line.strip().lstrip('#').strip()
                buf = []
            elif line.strip():
                buf.append(line)
    if name is not None and buf:
        sections[name] = pd.read_csv(io.StringIO(''.join(buf)))
    return sections
