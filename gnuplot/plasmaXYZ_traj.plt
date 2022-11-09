set terminal qt 0 size 800,800
unset key
set xrange [-0.05:0.05]
set yrange [-0.05:0.05]
set zrange [-0.05:0.05]
set grid
show grid

#FROM NT1 TO NT2 (every ::Ne*Nt1::Ne*Nt2)
Ne=0
Np=1000
Nt=Ne+Np
row=1
splot for [row=0:1000]'results.bin' binary format='%float%float%float' using 1:2:3 every ::(Nt*row*10+115)::(Nt*row*10+115) pointsize 0.3 pointtype 7 lt rgb "blue"

#set multiplot
#do for[row=0:1000] {
#splot 'results.bin' binary format='%float%float%float' using 1:2:3 every ::(Nt*row*10+1)::(Nt*row*10+1) #pointsize 0.3 pointtype 7 lt rgb "blue"
#}
#unset multiplot

