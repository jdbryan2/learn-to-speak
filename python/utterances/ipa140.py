import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa140 = ut.Utterance(dirname="ipa140",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa140.SetManualArticulation(aw.kArt_muscle.LUNGS,[0, 0.05, 0.5],[0.3, 0.2, 0.2])
ipa140.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI,[0, 0.5],[1, 1])
ipa140.SetManualArticulation(aw.kArt_muscle.MASSETER,[0.0, 0.5],[0.3, 0.3])
ipa140.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS,[0.0, 0.5],[0.5, 0.5])

ipa140.Run()