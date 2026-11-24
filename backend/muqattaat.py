"""v1.3: the disconnected letters that open 29 surahs (الم, كهيعص, طه, حم, ن ...).

The ASR already writes them in the same compact form the database holds, so no alias table is needed.
The one problem: a very short first partial can read الم even when the reciter said the WORD "alam" (the final pass
then says عَلَمْ), and a correct word is committed the moment it is heard. is_muq() lets the server require two agreeing
passes (or the final pass) before committing such a word."""

# (surah, ayah) whose FIRST word is a letters opening
MUQ = {(2, 1), (3, 1), (7, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (19, 1), (20, 1), (26, 1),
       (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (36, 1), (38, 1), (40, 1), (41, 1), (42, 1), (42, 2),
       (43, 1), (44, 1), (45, 1), (46, 1), (50, 1), (68, 1)}

def is_muq(surah, word) -> bool:
    return (surah, word["ayah_number"]) in MUQ and word["word_position"] == 1
