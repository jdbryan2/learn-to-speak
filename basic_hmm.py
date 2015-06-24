from hmm import HMM
import pylab as plt
import re


# load the text file into an array
f = open('cn_data.txt', 'r')

raw_text = f.read()
observation_sequence = re.sub('[\']+', '', raw_text.lower())
observation_sequence = re.sub('[^a-z]+', ' ', observation_sequence)
observation_sequence = re.sub('[ ]+', ' ', observation_sequence)


hmm = HMM(observation_sequence, 7);
hmm.Baum(0.01)
#print hmm.alpha
print hmm.beta
plt.plot(hmm.Prob_hist)
plt.show()

hmm.Generator(50)

plt.figure()
plt.hold(True)

p = []
leg = []
plot_color = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
plot_marker = ('o', 's', '^', '*', 'x', 'd')

clip = False
if clip:
	start_pos = 1
else:
	start_pos = 0

for j in range(hmm.N):
	p.append(plt.scatter(range(start_pos, len(hmm.V)), [hmm.B[j, k] for k in range(start_pos, len(hmm.V))], marker=plot_marker[0], color=plot_color[j]))
	leg.append(r'$b_{' + str(j+1) + ' k}$')
plt.hold(False)


plt.xlabel(r'$v_k$', fontsize=14)
plt.ylabel(r'$b_{j k}$', fontsize=14)
plt.xticks([x for x in range(len(hmm.V))], hmm.V)
plt.grid(True)
plt.legend(p, leg, loc='center left', bbox_to_anchor=(1, 0.5))	
plt.subplots_adjust(right=0.84, bottom=0.15)
plt.show()
