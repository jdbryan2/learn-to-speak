import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ipa142 = ut.Utterance(dirname="ipa142",
                loops=1,
                utterance_length=0.5,
                addDTS=False)

ipa142.SetManualArticulation(aw.kArt_muscle.LUNGS,[0, 0.05, 0.5],[0.3, 0.25, 0.22])
ipa142.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI,[0, 0.5],[1, 1])
ipa142.SetManualArticulation(aw.kArt_muscle.MYLOHYOID,[0.0, 0.5],[0.2, 0.2])
ipa142.SetManualArticulation(aw.kArt_muscle.SPHINCTER,[0.0, 0.5],[0.6, 0.6])

ipa142.Run()