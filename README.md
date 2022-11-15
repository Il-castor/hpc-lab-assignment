# hpc-lab-assignment-1

## Autori
* Balma Giovanni (320677)
* Castorini Francesco (274987)
* Rossi Lorenzo (273693)


## Eseguire il codice 
Entrare nella cartella assignment_01 e digitare il seguente comando per compilare le varie versioni di durbin sulla CPU: 

```console
make FILE=nome_file_senza_estensione EXT_CFLAGS="-DPOLYBENCH_TIME -DNTHREADS=4 -DLARGE_DATASET -DPRINT_HASH" clean run
```
Per compilare durbin sulla GPU digitare i seguenti comandi:

```console
module load clang/11.0.0 cuda/10.0
```
e dopo
```console
./compile_gpu.sh
```

