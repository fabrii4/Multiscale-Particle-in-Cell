set terminal qt 0 size 800,800
unset key
set xrange [-0.2:0.2]
set yrange [-0.2:0.2]
set zrange [-0.2:0.2]
set grid
show grid

#TXT FILE
#splot for [col=1:3003:3] 'results.txt' using col:col+1:col+2 pointsize 0.01 pointtype 7
#TXT FILE 2
#stats 'results.txt' using 0 nooutput
#splot for [i=0:(STATS_blocks - 2)] 'results.txt' using 1:2:3 index i pointsize 0.01 pointtype 7

#BIN FILE
#ALL POINTS
splot 'lost.bin' binary format='%float%float%float' using 1:2:3 pointsize 0.1 pointtype 7

#FROM N1 TO N2 (every ::Ne*N1::Ne*N2)
row=0
splot 'results.bin' binary format='%float%float%float' using 1:2:3 every ::0*row*10::1000*(row*10) pointsize 0.01 pointtype 7 


