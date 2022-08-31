#include <stddef.h>

/* Structs */

typedef struct Ht_item Ht_item;

// Define the Hash Table Item here
struct Ht_item {
    long key;
    int value;
};


typedef struct LinkedList LinkedList;

// Define the Linkedlist here
struct LinkedList {
    Ht_item* item; 
    LinkedList* next;
};

typedef struct HashTable HashTable;

// Define the Hash Table here
struct HashTable {
    // Contains an array of pointers
    // to items
    Ht_item** items;
    LinkedList** overflow_buckets;
    int size;
    int count;
};


/* FUNCTION DECLARATIONS FOR API */
HashTable* create_table(int size);
void ht_insert(HashTable* table, long key, int value);
int ht_search(HashTable* table, long key);
void free_table(HashTable* table);