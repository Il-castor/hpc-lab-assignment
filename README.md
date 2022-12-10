# hpc-lab-assignment

## Autori
* Balma Giovanni (320677)
* Castorini Francesco (274987)
* Rossi Lorenzo (273693)


## Eseguire il codice 
Entrare nella cartella `assignment` e digitare il seguente comando per compilare le varie versioni di durbin sulla CPU: 

```console
make FILE=nome_file.c EXT_CXXFLAGS="-DPOLYBENCH_TIME -DLARGE_DATASET -DPRINT_HASH" clean run
```
Per compilare durbin sulla GPU con OpenMP digitare i seguenti comandi:

```console
module load clang/11.0.0 cuda/10.0
./compile_gpu.sh
./durbin_gpu.out
```
Per compilare durbin sulla GPU con CUDA digitare i seguenti comandi: 

```console
make FILE=durbin_cuda.cu EXT_CXXFLAGS="-DLARGE_DATASET" clean run
```

## Assignment opzionale
Il codice opzionale si può trovare nella cartella `optional`, per eseguire il codice si può digitare il comando:

```console
make FILE=nome_file.c clean run
```
