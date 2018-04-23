import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa305 = ut.Utterance(dirname="ipa305",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa305.InitializeAll()
ipa305.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa305.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.05, 0.5],[0.3, 0.25, 0.26])
ipa305.SetManualArticulation(aw.kArt_muscle.MYLOHYOID, [0.0, 0.5], [0.1, 0.1])
ipa305.SetManualArticulation(aw.kArt_muscle.SPHINCTER, [0.0, 0.5], [0.7, 0.7])
ipa305.SetManualArticulation(aw.kArt_muscle.HYOGLOSSUS, [0.0, 0.5], [0.3, 0.3])

ipa305.Run()
