import primitive.Utterance as ut
import Artword as aw

click = ut.Utterance(dirname="click",
                  loops=1,
                  utterance_length=0.5)
    
click.SetManualArticulation(aw.kArt_muscle.MASSETER, [0.0 , 0.2 ,  0.3 ,  0.5 ],[0.25, 0.25, -0.25, -0.25])
click.SetManualArticulation(aw.kArt_muscle.ORBICULARIS_ORIS, [0.0 , 0.2 , 0.3, 0.5],[0.75, 0.75, 0.0, 0.0])
click.SetManualArticulation(aw.kArt_muscle.STYLOGLOSSUS, [0.0, 0.5],[0.9, 0.9])

click.Run()