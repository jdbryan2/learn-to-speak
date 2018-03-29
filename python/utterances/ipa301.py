import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa301 = ut.Utterance(dirname="ipa301",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa301.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa301.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5],[1.0, 1.0])
ipa301.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.05, 0.5],[0.3, 0.2, 0.2])
ipa301.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.0, 0.5], [0.1, 0.1])
ipa301.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.5], [0.95, 0.95])
ipa301.SetManualArticulation(aw.kArt_muscle.GENIOGLOSSUS, [0.0, 0.5], [1.0, 1.0])

ipa301.Run()