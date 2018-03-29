import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
sigh = ut.Utterance(dirname="sigh",
                   loops=1,
                   utterance_length=0.5)

sigh.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1],[0.1, 0.0])
sigh.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5],[1.0, 1.0])

sigh.Run()