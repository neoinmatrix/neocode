from thinkbayes import Pmf

# pmf = Pmf()
# pmf.Set('Bowl 1', 0.5)
# pmf.Set('Bowl 2', 0.5)

# pmf.Mult('Bowl 1', 0.75)
# pmf.Mult('Bowl 2', 0.5)

# pmf.Normalize()

# print pmf.Prob('Bowl 1')
# wordlist=["neo","matrix","human"]
# for word in wordlist:
# 	pmf.Incr(word,1)
# pmf.Incr("neo",1)
# pmf.Normalize()
# pmf.Print()

class Cooke(Pmf):
	def __init__(self,hypos):
		Pmf.__init__(self)
		for hypo in hypos:
			self.Set(hypo,1)
		self.Normalize()

	def Update(self,data):
		for hypo in self.Values():
			like=self.Likelihood(data,hypo)
			self.Mult(hypom,like)
		self.Normalize()

	def Likelihood(self,data,hypo):
		mix=self.mixes[hypo]
		like=mix(data)
		return like

c=Cooke(['bowl1','bowl2'])
c.Print()
mix=['bowl3','bowl4']
c.Update(mix)
c.Print()