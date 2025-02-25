
from pathlib import Path

import numpy as np


class AlignmentExtractor:
    def __init__(self):
        pass

    def align_from_sample(self, sample):
        """
        Creates alignment indices for source-target token pairs.
        
        For Japanese text conversion (kana to kanji), this maps positions where
        source characters (kana) combine to form target tokens (kanji/compounds).
        
        Each alignment index represents:
        - The ending position (1-based) in the source sequence where a target token is complete
        - For example, if source "こ れ" maps to target "これ", alignment will be "2"
        because "これ" is complete after the 2nd source character
        
        Args:
            sample: Tuple of (source_tokens, target_tokens)
                source_tokens: List of individual characters (e.g., kana)
                target_tokens: List of compound tokens (e.g., kanji)
        
        Returns:
            String of space-separated alignment indices, with two "9999" padding values appended
            Example: For source="こ れ ま た" and target="これ また", returns "2 4 9999 9999"
        """
        source, target = sample
        
        # Track cumulative lengths and token boundaries
        current_pos = 0
        alignment = []
        source_pos = 0
        
        for tgt_token in target:
            # For each target token, count how many source characters it uses
            token_len = len(tgt_token)
            source_pos += token_len
            # Record the position where this target token is complete
            alignment.append(source_pos)
        
        # Add two padding values (9999) as required by the model
        alignment = alignment + [9999, 9999]
        
        assert len(target) > 0
        assert len(alignment) == len(target) + 2, f"{source} != {target} + 2"
        
        # Convert to space-separated string format
        alignment = " ".join([str(item) for item in alignment])
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
