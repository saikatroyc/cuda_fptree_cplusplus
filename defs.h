#ifdef TEST_MODE
    #define max_items_in_transaction 10
    #define max_num_of_transaction 24//00000
    #define max_unique_items 30//32000
    #define BLOCK_SIZE 1024
    #define TRANSACTION_PER_SM 4
    #define support 1
#else
    #define max_items_in_transaction 64//128//64//32
    #define max_num_of_transaction 1000000
    #define max_unique_items 32000
    #define BLOCK_SIZE 1024
    #define TRANSACTION_PER_SM 192//96//47//95//380//190//
    #define support 1
#endif
#define INVALID 0XFFFFFF 
