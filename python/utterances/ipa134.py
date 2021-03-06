import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa134 = ut.Utterance(dirname="ipa134",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa134.SetManualArticulation(aw.kArt_muscle.LUNGS,[0, 0.05, 0.5],[0.3, 0.2, 0.2])
ipa134.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI,[0, 0.5],[1, 1])
ipa134.SetManualArticulation(aw.kArt_muscle.MASSETER,[0.0, 0.5],[0.35, 0.35])
ipa134.SetManualArticulation(aw.kArt_muscle.GENIOGLOSSUS,[0.0, 0.5],[0.2, 0.2])
ipa134.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS,[0.0, 0.5],[0.6, 0.6])

ipa134.Run()