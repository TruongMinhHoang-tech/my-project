import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
sys.setrecursionlimit(2000000)
plt.rcParams['font.family'] = 'Arial'

np.random.seed(13)
def generate_dataset():
 N = 1000000
 dataset = {}

 dataset['1_int_ascending'] = np.arange(1, N + 1, dtype=np.int32)
 num_float = np.linspace(1.0, 100000.0, N, dtype=np.float32)
 dataset['2_float_descending'] = num_float[::-1].copy()

 for i in range(3, 7):
    dataset[f'{i}_int_random'] = np.random.randint(1, 10000000, size=N, dtype=np.int32)
 for i in range(7, 11):
    dataset[f'{i}_float_random'] = np.random.uniform(1.0, 100000.0, size=N).astype(np.float32)
 return dataset

#Quick sort
def partition(arr, low, high):
    mid = (low + high) // 2
    arr[mid], arr[high] = arr[high], arr[mid]
    pivot = arr[high]

    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def call_quicksort(arr):
    quick_sort(arr, 0, len(arr) - 1)


#Merge sort
def call_mergesort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        Left = arr[:mid].copy()
        Right = arr[mid:].copy()

        call_mergesort(Left)
        call_mergesort(Right)

        i = j = k = 0
        while i < len(Left) and j < len(Right):
            if Left[i] <= Right[j]:
                Left[i] = arr[k]
                i+=1
            else:
                Right[j] = arr[k]
                j+=1
            k+=1

        while i < len(Left):
            arr[k] = arr[i]
            i+=1
            k+=1
        while j < len(Right):
            arr[k] = arr[j]
            j+=1
            k+=1


#Heap sort
def heapify(arr, n, i):
    largest = i
    l = 2*i+1
    r = 2*i+2
    
    if l < n and arr[l] > arr[largest]:
        largest = l
    if r < n and arr[r] > arr[largest]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def call_heapsort(arr):
     n = len(arr)
     for i in range(n // 2 - 1, -1, -1):
         heapify(arr, n, i)
     for i in range(n - 1, 0, -1):
         arr[0], arr[i] = arr[i], arr[0]
         heapify(arr, i, 0)





def run_experiments():
    
    datasets = generate_dataset()
    
    algorithms = {
        "QuickSort": call_quicksort,
        "HeapSort": call_heapsort,
        "MergeSort": call_mergesort,
        "Numpy_Sort": np.sort 
    }
    
    results = [] 
    
    for data_name, data_arr in datasets.items():
        row_result = {"Data name": data_name}
       
        for algo_name, algo_func in algorithms.items():
            arr_test = data_arr.copy()
           
            start_time = time.perf_counter()
            
           
            algo_func(arr_test)
                
            
            end_time = time.perf_counter()
            
            time_taken = end_time - start_time
            row_result[algo_name] = time_taken
            print(f"{algo_name:<12}: {time_taken:.4f} giây")
            
        results.append(row_result)
        
    
    df = pd.DataFrame(results)
    df.to_csv("Ket_qua_thuc_nghiem.csv", index=False)
    print(df.to_string())
    return df

def plot_results(df):
    
    
    df.set_index('Data name').plot(kind='bar', figsize=(14, 7), width=0.8)
    
    plt.title('So sánh thời gian chạy 4 thuật toán sắp xếp', fontsize=16)
    plt.ylabel('Thời gian thực nghiệm (s)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    
    plt.savefig('Bieu_do_thoi_gian.png')
    plt.show()

if __name__ == "__main__":
    
    df_results = run_experiments()
    
    plot_results(df_results)
