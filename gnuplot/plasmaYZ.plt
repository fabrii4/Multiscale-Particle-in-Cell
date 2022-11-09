set terminal qt 0 size 800,800
unset key
set multiplot
set xrange [0:799] 
set yrange [0:799] 
unset tics
unset border
plot 'BfieldYZ3.png' binary filetype=png with rgbimage
set xrange [-0.2:0.223]
set yrange [-0.21:0.22]
set border
set tics
set grid
show grid

#TXT FILE
#plot for [col=1:3003:3] 'results.txt' using col+1:col+2 pointsize 0.01 pointtype 7
#TXT FILE 2
#stats 'results.txt' using 0 nooutput
#plot for [i=0:(STATS_blocks - 2)] 'results.txt' using 2:3 index i pointsize 0.01 pointtype 7

#BIN FILE
#ALL POINTS
#plot 'results.bin' binary format='%float%float%float' using 1:2 pointsize 0.1 pointtype 7 

#FROM NT1 TO NT2 (every ::Ne*Nt1::Ne*Nt2)
Ne=1
Np=999
Nt=Ne+Np
row=10
plot 'results.bin' binary format='%float%float%float' using 2:3 every ::((Nt*row*10)+0)::(Nt*(row*10)+Ne) pointsize 0.3 pointtype 7 lt rgb "red"
plot 'results.bin' binary format='%float%float%float' using 2:3 every ::((Nt*row*10)+Ne)::(Nt*(row*10)+Nt) pointsize 0.3 pointtype 7 lt rgb "blue"

#plot 'lost.bin' binary format='%float%float%float' using 2:3 every ::1::1000 pointsize 0.5 #pointtype 7 lt rgb "green"
unset multiplot
