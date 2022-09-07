#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hash_table.h"


/* IMPLEMENTATION OF HASH TABLE FROM: https://www.digitalocean.com/community/tutorials/hash-table-in-c-plus-plus */

// changed the types of key, value to (long, int)

/* WILL USE HASH TABLE FOR LOOKUP FROM address delta -> encoding index */


#define CAPACITY 50000 // Size of the Hash Table

// found at "https://lemire.me/blog/2018/08/15/fast-strongly-universal-64-bit-hashing-everywhere/"
long hash_function(long h)
{
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53L;
  h ^= h >> 33;
  return (h % CAPACITY);
}

static LinkedList* allocate_list () {
    // Allocates memory for a Linkedlist pointer
    LinkedList* list = (LinkedList*) malloc (sizeof(LinkedList));
    return list;
}

static LinkedList* linkedlist_insert(LinkedList* list, Ht_item* item) {
    // Inserts the item onto the Linked List
    if (!list) {
        LinkedList* head = allocate_list();
        head->item = item;
        head->next = NULL;
        list = head;
        return list;
    } 
    
    else if (list->next == NULL) {
        LinkedList* node = allocate_list();
        node->item = item;
        node->next = NULL;
        list->next = node;
        return list;
    }

    LinkedList* temp = list;
    while (temp->next->next) {
        temp = temp->next;
    }
    
    LinkedList* node = allocate_list();
    node->item = item;
    node->next = NULL;
    temp->next = node;
    
    return list;
}

// apparaently not used (compiler said so, thus commented out to make it happy..)

// static Ht_item* linkedlist_remove(LinkedList* list) {
//     // Removes the head from the linked list
//     // and returns the item of the popped element
//     if (!list)
//         return NULL;
//     if (!list->next)
//         return NULL;
//     LinkedList* node = list->next;
//     LinkedList* temp = list;
//     temp->next = NULL;
//     list = node;
//     Ht_item* it = NULL;
//     memcpy(temp->item, it, sizeof(Ht_item));
//     free(temp->item);
//     free(temp);
//     return it;
// }

static void free_linkedlist(LinkedList* list) {
    LinkedList* temp = list;
    while (list) {
        temp = list;
        list = list->next;
        free(temp->item);
        free(temp);
    }
}

static LinkedList** create_overflow_buckets(HashTable* table) {
    // Create the overflow buckets; an array of linkedlists
    LinkedList** buckets = (LinkedList**) calloc (table->size, sizeof(LinkedList*));
    for (int i=0; i<table->size; i++)
        buckets[i] = NULL;
    return buckets;
}

static void free_overflow_buckets(HashTable* table) {
    // Free all the overflow bucket lists
    LinkedList** buckets = table->overflow_buckets;
    for (int i=0; i<table->size; i++)
        free_linkedlist(buckets[i]);
    free(buckets);
}


Ht_item* create_item(long key, int value) {
    // Creates a pointer to a new hash table item
    Ht_item* item = (Ht_item*) malloc (sizeof(Ht_item));
    item->key = key;
    item->value = value;

    return item;
}

HashTable* create_table(int size) {
    // Creates a new HashTable
    HashTable* table = (HashTable*) malloc (sizeof(HashTable));
    table->size = size;
    table->count = 0;
    table->items = (Ht_item**) calloc (table->size, sizeof(Ht_item*));
    for (int i=0; i<table->size; i++)
        table->items[i] = NULL;
    table->overflow_buckets = create_overflow_buckets(table);

    return table;
}

void free_item(Ht_item* item) {
    // Frees an item
    free(item);
}

void free_table(HashTable* table) {
    // Frees the table
    for (int i=0; i<table->size; i++) {
        Ht_item* item = table->items[i];
        if (item != NULL)
            free_item(item);
    }

    free_overflow_buckets(table);
    free(table->items);
    free(table);
}

void handle_collision(HashTable* table, long index, Ht_item* item) {
    LinkedList* head = table->overflow_buckets[index];

    if (head == NULL) {
        // We need to create the list
        head = allocate_list();
        head->item = item;
        table->overflow_buckets[index] = head;
        return;
    }
    else {
        // Insert to the list
        table->overflow_buckets[index] = linkedlist_insert(head, item);
        return;
    }
 }

void ht_insert(HashTable* table, long key, int value) {
    // Create the item
    Ht_item* item = create_item(key, value);

    // Compute the index
    long index = hash_function(key);

    Ht_item* current_item = table->items[index];
    
    if (current_item == NULL) {
        // Key does not exist.
        if (table->count == table->size) {
            // Hash Table Full
            printf("Insert Error: Hash Table is full\n");
            // Remove the create item
            free_item(item);
            return;
        }
        
        // Insert directly
        table->items[index] = item; 
        table->count++;
    }

    else {
            // Scenario 1: We only need to update value
            if (current_item->key == key) {
                table->items[index]->value = value;
                return;
            }
    
        else {
            // Scenario 2: Collision
            handle_collision(table, index, item);
            return;
        }
    }
}

int ht_search(HashTable* table, long key) {
    // Searches the key in the hashtable
    // and returns NULL if it doesn't exist
    long index = hash_function(key);
    Ht_item* item = table->items[index];
    LinkedList* head = table->overflow_buckets[index];

    // Ensure that we move to items which are not NULL
    while (item != NULL) {
        if (item->key == key)
            return item->value;
        if (head == NULL)
            return -1;
        item = head->item;
        head = head->next;
    }
    return -1;
}

void print_search(HashTable* table, long key) {
    int val;
    if ((val = ht_search(table, key)) == -1) {
        printf("%ld does not exist\n", key);
        return;
    }
    else {
        printf("Key:%ld, Value:%i\n", key, val);
    }
}

void print_table(HashTable* table) {
    printf("\n-------------------\n");
    for (unsigned long i=0; i<table->size; i++) {
        if (table->items[i]) {
            printf("Index:%lu, Key:%ld, Value:%i", i, table->items[i]->key, table->items[i]->value);
            if (table->overflow_buckets[i]) {
                printf(" => Overflow Bucket => ");
                LinkedList* head = table->overflow_buckets[i];
                while (head) {
                    printf("Key:%ld, Value:%i ", head->item->key, head->item->value);
                    head = head->next;
                }
            }
            printf("\n");
        }
    }
    printf("-------------------\n");
}

// int main() {
//     HashTable* ht = create_table(CAPACITY);
//     ht_insert(ht, 1, 10);
//     ht_insert(ht, 2, 11);
//     ht_insert(ht, 3, 12);
//     ht_insert(ht, -4, 13);
//     print_search(ht, 1);
//     print_search(ht, 2);
//     print_search(ht, 3);
//     print_search(ht, 4);
//     print_search(ht, -4);
//     ht_insert(ht, 4, 14);
//     print_search(ht, 4);
//     print_search(ht, 5);
//     print_table(ht);
//     free_table(ht);
//     return 0;
// }
