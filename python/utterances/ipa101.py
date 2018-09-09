import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa101 = ut.Utterance(dirname="../data/utterances/ipa101",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa101.InitializeAll()

ipa101.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0, 0.5],[0.5, 0.5])
ipa101.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5],[1.0, 1.0])
ipa101.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1],[0.2, 0.0])
ipa101.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.25], [0.7])
ipa101.SetManualArticulation(aw.kArt_muscle.ORBICULARIS_ORIS, [0.25], [0.2])

ipa101.Run()
