# hpc-lab-assignment-1

## Autori
* Balma Giovanni (320677)
* Castorini Francesco (274987)
* Rossi Lorenzo (273693)


## Eseguire il codice 
Entrare nella cartella `assignment_01` e digitare il seguente comando per compilare le varie versioni di durbin sulla CPU: 

```console
make FILE=nome_file.c EXT_CFLAGS="-DPOLYBENCH_TIME -DLARGE_DATASET -DPRINT_HASH" clean run
```
Per compilare durbin sulla GPU digitare i seguenti comandi:

```bash
module load clang/11.0.0 cuda/10.0
./compile_gpu.sh
./durbin_gpu.out
```

## Assignment opzionale
Il codice opzionale si può trovare nella cartella `optional_01`, per eseguire il codice si può digitare il comando:

```console
make FILE=nome_file.c clean run
```
