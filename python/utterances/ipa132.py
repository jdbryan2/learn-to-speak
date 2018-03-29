import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa132 = ut.Utterance(dirname="ipa132",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa132.SetManualArticulation(aw.kArt_muscle.LUNGS,[0, 0.05, 0.5],[0.4, 0.3, 0.3])
ipa132.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI,[0, 0.5],[1, 1])
ipa132.SetManualArticulation(aw.kArt_muscle.MASSETER,[0.0, 0.5],[0.35, 0.35])
ipa132.SetManualArticulation(aw.kArt_muscle.GENIOGLOSSUS,[0.0, 0.5],[0.8, 0.8])
ipa132.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS,[0.0, 0.5],[1, 1])

ipa132.Run()