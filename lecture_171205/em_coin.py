#-*- coding: utf-8 -*-
import math
from collections import Counter
observed_data = ['HTTTHHTHTH','HHHHTHHHHH',
'HTHHHHHTHH','HTHTTTHHTT','THHHTHHHTH']

observed_data = [[x for x in i] for i in observed_data]

def getPosterior(thetaA,thetaB,heads,tails,verbose=False):
    new_ThetaA = math.pow(thetaA, heads) * math.pow((1-thetaA), tails)
    new_ThetaB = math.pow(thetaB, heads) * math.pow((1-thetaB), tails)
    conditional_thetaA = new_ThetaA / (new_ThetaA + new_ThetaB)
    conditional_thetaB = new_ThetaB / (new_ThetaA + new_ThetaB)
    coinA = (round(conditional_thetaA*heads,2),round(conditional_thetaA*tails,2))
    coinB = (round(conditional_thetaB*heads,2),round(conditional_thetaB*tails,2))
    if verbose:
        print u'Estimated θ(A):%f , estimated θ(B):%f' % (conditional_thetaA,conditional_thetaB)

    return conditional_thetaA,conditional_thetaB,coinA[0],coinA[1],coinB[0],coinB[1]

if __name__ == '__main__':

    thetaA,thetaB = 0.6,0.5 # One can play with this initial parameters around and get the same outcome (under thetaA >= thetaB)
    iterations = 10
    print u'Initial parameters : θ(A)=%f θ(B)=%f \n' % (thetaA,thetaB)

    for i in xrange(1,iterations+1):
        coinAheads_total,coinAtails_total = 0,0
        coinBheads_total,coinBtails_total = 0,0

        for datapoint in observed_data:
            datapoint = Counter(datapoint)

            heads = datapoint.get('H')
            tails = datapoint.get('T')

            if i<=2:
                _,_,coinAheads,coinAtails,coinBheads,coinBtails = getPosterior(thetaA, thetaB, heads, tails,False)
            else:
                _,_,coinAheads,coinAtails,coinBheads,coinBtails = getPosterior(thetaA, thetaB, heads, tails)

            coinAheads_total += coinAheads
            coinAtails_total += coinAtails
            coinBheads_total += coinBheads
            coinBtails_total += coinBtails

        thetaA = coinAheads_total / (coinAheads_total+coinAtails_total)
        thetaB = coinBheads_total / (coinBheads_total+coinBtails_total)
        print u'Round %i : Estimated θ(A):%f , estimated θ(B):%f \n' % (i,thetaA,thetaB)