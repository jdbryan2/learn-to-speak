import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa316 = ut.Utterance(dirname="../data/utterances/ipa316",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa316.InitializeAll()
ipa316.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa316.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.05, 0.5],[0.3, 0.25, 0.25])
ipa316.SetManualArticulation(aw.kArt_muscle.MYLOHYOID, [0.0, 0.5], [0.1, 0.1])
ipa316.SetManualArticulation(aw.kArt_muscle.SPHINCTER, [0.0, 0.5], [0.3, 0.3])
ipa316.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.5], [0.3, 0.3])

ipa316.Run()
