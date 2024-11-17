idev -N1
cd distVC2/
module load cuda/10.2
module load gcc/7.3.0
nvcc -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm kernel.cu -o op_testNew
old cmds:
ibrun -np 4 ./op_testNew
ibrun -gpu -np 4 ./op_testNew
ibrun -gpu -np 4 ./op_testNew roadNet-CA_wp_sor color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50

latest cmds:
ibrun -gpu -np 4 ./op_DistColor2hop_LargeColor roadNet-CA color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50_withoutWT partitionAll.txt

****now it takes graph without wt





ALL STEPS:

Using PULP:
Step 1:Change our format graph to Metis format:
g++ -o op_toMetis CovertToMetisFormat.cpp
./op_toMetis graph.txt 4 > graph_metis


Step 2: partitioning (for more about partitioning using prev. partitions see github)
make pulp
OR>>> 
g++ -fopenmp -O3 -Wall -c pulp.cpp
g++ -fopenmp -O3 -Wall -o pulp pulp_main.cpp pulp.o

****The write_parts function in io.cpp is modified to generate multiple partition files containing vertex id for each partitions
file names are "partition<partition id>.txt"
Run on Foundry. Forge will not work due to lower gcc version. ****On desktop it does not work properly****
Use label propagation to get better partition(-l):
./pulp graphSample.txt [no. of partition] -l

(e.g. ./pulp roadNet-CA_Metis 4 -l)it creates 4 partition files and another file with all partitionID 


Alternative:(use xtraPULP. partition in foundry):
-----------------
sinteractive -p cuda --time=04:00:00 --gres=gpu:4 --nodes=4 --mem=32000M
or use multiple CPUs using --ntasks

module load openmpi/4.0.3/gnu/9.2.0

convert to bin format for xtraPULP:
./op_edgeToBin graphTest.txt graphOutBin.txt

partition using xtraPULP:(for some reason unable to use multiple CPUs. so use -n 1)
mpirun -n 1 ./xtrapulp <binary file> 2 -o partitionOutput.2parts





Step 3: Generate initial colors using Gunrock (single GPU)
to run and print only colors:
./test_color_11.0_x86_64_printcolor market bips98_606.mtx --undirected=1 --quiet > colorFile

Alternative(generate color using kokkos kernel):
------------------------------------------------
convert graph to mtx format
e.g.: ./op_toMtx graph [# of nodes] > graph.mtx
Run::
./graph_color --openmp 8 --algorithm COLORING_EB --amtx test_GraphAK.mtx --outputfile result

*****Important*****
input file format for kokkos should be .mtx format with first line "%%MatrixMarket matrix coordinate real general" if we have "a b" and "b a" both in the file.
else first line should be first line "%%MatrixMarket matrix coordinate real symmetric" 
If this 1st line is not there, the code will not work.
The input file should have .mtx written after file name, else it will not work

Change the color output to our color format:
e.g.: ./op_KokkoscolorFormatToOurColor color_RMAT24_e2_kokkos > color_RMAT24_e2_kokkos_our


step 4:
latest cmds to run our code:
ibrun -gpu -np 4 ./op_DistColor2hop_LargeColor roadNet-CA color_roadNet-CA_wp_sor roadNet-CA_cE_100K_50_withoutWT partitionAll.txt
