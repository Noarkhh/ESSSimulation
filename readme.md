# Simulation of animal behavior patterns in context of evolutionary stable strategies.

Evolutionary Stable Strategy (or ESS; also evolutionarily stable strategy): A behavioral strategy (phenotype) if adopted 
by all individuals in a population that cannot be replaced or invaded by a different strategy through natural 
selection.[^1] In game-theoretical terms, an ESS is an equilibrium refinement of the Nash equilibrium, being a Nash 
equilibrium that is also "evolutionarily stable."[^2]

When two animals contest, the outcome is determined by their behavior type. Only one animal can get the prize, 
wasting time and getting injured are punished. The simplified point system[^3] is as follows:
- Winning: 50 points
- Losing: 0 points
- Injury: -100 points
- Wasting Time: -10 points

For now let's consider only two behaviors, Dove and Hawk. The Dove will spend time contesting the prize non-violently (posing etc.) until its opponent
gives up. If attacked immediately flees leaving the prize behind but avoiding injury. The Hawk will fight for the prize until victorious or injured.
With these behaviors defined we can get the following table of expected average outcomes:

|        |Dove|Hawk|    
|--------|:--:|:--:|
|**Dove**|15  |  0 |
|**Hawk**|50  | -25|
 
Value from each cell represents how much on average will a behavior from its' row get when confronted with another behavior from its' column.
For example: A dove when contesting with another dove will on average earn 15 points each, because they will both waste time 
$(2 \times -10)$ and only one of them will get the prize $(+50)$, this gives average value of $\dfrac{2 \times -10 + 50}{2} = 15$.
 
Population consisting of only doves and hawks will reach an equilibrium with the ratio 5:7 (doves:hawks) (Figure 1.1). The same ratio can be achieved
by solving a system of equations based on the outcome table with d and h variables and a parameter n:
 
$$
\begin{cases}
15 \times d + 0 \times h = n\\
50 \times d - 15 \times h = n\\
\end{cases}
$$
 
This means that neither hawks or doves will gain advantage over the other. We also don't assume, that an animal can act in only one way,
so acting like a dove with probability $\dfrac{5}{12}$ and acting like a hawk with probability $\dfrac{7}{12}$ could be considered 
an **Evolutionary Stable Strategy**

Let's introduce another 3 behaviors:
- Retaliator: Will act like a dove until attacked, then it will retaliate.
- Bully: Will act like a hawk until getting attacked, then will act like a dove.
- Prober-Retaliator: Will act like a retaliator, but sometimes probes the contestant by attacking.

We will also use the advanced version of outcome table[^4], for it has been studied more and is more complex 
(we will add 100 to each cell for simulation purposes):

|                     |Dove |Hawk |Retaliator|Bully|Prober-Retaliator|    
|---------------------|:---:|:---:|:--------:|:---:|:---------------:|
|**Dove**             |129  |119.5|129       |119.5|117.2            |
|**Hawk**             |180  |80.5 |81.9      |174.6|81.10            |
|**Retaliator**       |129  |77.7 |129       |157.1|123.1            |
|**Bully**            |180  |104.9|111.9     |141.5|111.2            |
|**Prober-Retaliator**|156.7|79.9 |126.9     |159.4|121.9            |

And from now on, instead of table we will use the **outcome matrix** with the ordering like in the table above:

$$
\left(\begin{array}{cc} 
129 & 119.5 & 129 & 119.5 & 117.2\\
180 & 80.5 & 81.9 & 174.6 & 81.10\\  
129 & 77.7 & 129 & 157.1 & 123.1\\  
180 & 104.9 & 111.9 & 141.5 & 111.2\\
156.7 & 79.9 & 126.9 & 159.4 & 121.9\\
\end{array}\right)
$$
 
[^1]: Cowden, C. C. (2012). *Game Theory, Evolutionary Stable Strategies and the Evolution of Biological Interactions.* Nature Education Knowledge 3(10):6.

[^2]: Evolutionarily stable strategy. *Wikipedia*. https://en.wikipedia.org/wiki/Evolutionarily_stable_strategy.

[^3]: Dawkins, R. (1989). *The Selfish Gene (Anniversary edition)*. Oxford University Press.

[^4]: Maynard Smith, J., Price, G. (1973). *The Logic of Animal Conflict.* Nature 246, pp. 15–18.

[^5]: Gale, J., Eaves, L. (1975). *Logic of animal conflict.* Nature 254, pp. 463.
