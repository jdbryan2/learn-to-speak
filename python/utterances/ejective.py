import primitive.Utterance as ut
import Artword as aw

# Default initial_art is all zeros
ejective = ut.Utterance(dirname="ejective",
                     loops=1,
                     utterance_length=0.5)
    
ejective.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1],[0.1, 0.0])
ejective.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI, [0.0, 0.5],[1.0, 1.0])
ejective.SetManualArticulation(aw.kArt_muscle.INTERARYTENOID, [0.0, 0.17, 0.2, 0.35, 0.38, 0.5],[0.5, 0.5 , 1.0, 1.0 , 1.0 , 0.5])
ejective.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.0, 0.5],[-.3, -.3])
ejective.SetManualArticulation(aw.kArt_muscle.HYOGLOSSUS, [0.0, 0.5],[0.5, 0.5])
ejective.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.1, 0.15, 0.29, 0.32],[0.0, 0.0, 1.0 , 1.0 , 0.0 ])
ejective.SetManualArticulation(aw.kArt_muscle.STYLOHYOID, [0.0, 0.22, 0.27, 0.35, 0.38, 0.5],[0.0,  0.0, 1.0 , 1.0 , 0.0 , 0.0])
    
ejective.Run()