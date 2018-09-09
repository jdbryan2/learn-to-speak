import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa304 = ut.Utterance(dirname="../data/utterances/ipa304",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa304.InitializeAll()
ipa304.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa304.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.5],[0.3, 0.22])
ipa304.SetManualArticulation(aw.kArt_muscle.MYLOHYOID, [0.0, 0.5], [0.1, 0.1])
ipa304.SetManualArticulation(aw.kArt_muscle.SPHINCTER, [0.0, 0.5], [0.7, 0.7])
ipa304.SetManualArticulation(aw.kArt_muscle.HYOGLOSSUS, [0.0, 0.5], [1.0, 1.0])
ipa304.SetManualArticulation(aw.kArt_muscle.GENIOGLOSSUS, [0.0, 0.5], [1.0, 1.0])

ipa304.Run()
