from pathlib import Path

import numpy as np


class AlignmentExtractor:
    def __init__(self):
        pass

    def align_from_sample(self, sample):
        """
        Generates alignment between kana characters and kanji tokens.
        The alignment value for each kanji token represents the position
        of the last kana character that corresponds to it.
        
        For example:
        kana:  こ の じ て ん で わ れ わ れ は
        kanji: この 時点 で われわれ は
        align: 1    4    5   9        10
        
        Output includes two right paddings (9999).
        """
        source, target = sample  # source: kana chars, target: kanji tokens
        
        current_pos = 0
        alignment = []
        
        # Create mapping from kanji tokens to their kana lengths
        kana_pos = 0
        for kanji_token in target:
            kana_length = 0
            while kana_pos < len(source) and kana_length < len(kanji_token):
                kana_pos += 1
                kana_length += 1
            alignment.append(kana_pos - 1)  # -1 because we want the last position
        
        # Add padding tokens
        alignment.extend([9999, 9999])
        
        # Convert to string format
        alignment = " ".join(str(item) for item in alignment)
        
        return alignment

    def align_from_file(self, src_file_path, tgt_file_path):
        src_file_path = Path(src_file_path)
        with open(src_file_path, "r", encoding="utf-8") as src_inp_file, open(
            tgt_file_path, "r", encoding="utf-8"
        ) as tgt_inp_file, open(
            src_file_path.with_suffix(".align"), "w", encoding="utf-8"
        ) as out_file:
            for src_line, tgt_file in zip(src_inp_file, tgt_inp_file):
                src_line = src_line.strip().split(" ")
                tgt_line = tgt_file.strip().split(" ")
                out_file.write(self.align_from_sample((src_line, tgt_line)) + "\n")


if __name__ == "__main__":
    # Example run:
    # python preprocess/mec2alignment.py data/train.kana data/train.kanji
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("kana_path", help="Path to the kana file")
    parser.add_argument("kanji_path", help="Path to the kanji file")
    args = parser.parse_args()

    exctractor = AlignmentExtractor()
    exctractor.align_from_file(args.kana_path, args.kanji_path)