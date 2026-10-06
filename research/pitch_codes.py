# Pitch codes from https://www.retrosheet.org/eventfile.htm ("The pitches field of the play record").
BALLS = set("BIPV")  # ball, intentional ball, pitchout, automatic/called ball (V)
STRIKES = set("ACKMQSTO")  # automatic strike, called, unknown strike, missed bunt, swing on pitchout, swinging, foul tip, foul tip on bunt
FOULS = set("FR")  # foul, foul on pitchout: a strike only with fewer than 2 strikes
FOUL_BUNT = "L"  # always a strike, so a foul bunt with 2 strikes is strike three
ENDS = set("HXY")  # hit batter, ball in play, ball in play on pitchout
UNKNOWN = "U"  # unknown or missed pitch: a pitch, effect on the count unknown
PITCHES = BALLS | STRIKES | FOULS | {FOUL_BUNT} | ENDS | {UNKNOWN}
NOT_PITCHES = set("+*.123>N")  # catcher pickoff, blocked, play not involving batter, pickoffs, runner going, no pitch
NOT_THROWN = set("VA")  # automatic ball/strike: change the count but are not thrown, and nump does not count them
DOCUMENTED = PITCHES | NOT_PITCHES


def count_after(seq):
    # Balls and strikes after every pitch in seq, with fouls not adding a third strike.
    balls = strikes = 0
    for ch in seq:
        if ch in BALLS:
            balls += 1
        elif ch in STRIKES or ch == FOUL_BUNT:
            strikes += 1
        elif ch in FOULS and strikes < 2:
            strikes += 1
    return balls, strikes
