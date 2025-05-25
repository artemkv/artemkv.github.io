# Relational Databases
{:.no_toc}

- A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
  {:toc}

TODO: this is work in progress

## Data normalization

- The main goal of normalization is eliminating redundancy
- The normalization is achieved by applying a sequence of rules to create normal forms
- A database is considered fully normalized when it is in fifth normal form
- Normal forms, informal: _every non-key attribute is dependent on the key, the whole key, and nothing but the key_ (so help me Codd)
- **1NF**: all columns are atomic (no multivalued columns)
- **2NF**: every non-key column is functionally dependent on the entire key (cannot depend on a part of the key only)
- Example of breaking the rule: [flight number, date, airline]. Flight number and the date uniquely identify the flight, making it a key. However, airline only depends on the flight number, not date
- **3NF**: every non-key column is non-transitively dependent on every key (non-key columns must be mutually independent)
- Example of breaking the rule: [flight number, airline code, airline name]. Flight number is a key, but airline code and airline name are not mutually independent: from airline code you can transitively find airline name
- Violations of fourth and fifth normal forms are very rare
- In some cases, deliberate denormalization is done to achieve better performance; however, this should be done only when there is a very good reason to do so

## Buffer pool

- The data is persistently stored on disk in the **data files**
- You cannot read or write a single byte to disk. Both reading and writing is performed in minimal units
- For HDD, the smallest unit of storage is a sector, size ranging from 512 bytes to 4K. For SSD, the smallest unit of storage is a page, typically 4K in size
- The relational databases are all designed to align with how the disks operate. This is why they also manage their memory in minimal units, called **pages** or **blocks**
- The typical size of a page is 8K (SQL Server, PostgreSQL, with Oracle allowing pages to be 2K, 4K, 8K or 16K)
- The pages need to be read into memory to be usable. **Buffer pool** (or **buffer cache**) is an area in main memory where databases store pages read from disk. Each **buffer** can fit exactly one page (plus page header)
- The amount of data stored on disk can largely exceed the available memory
- So the buffer pool acts as a cache for most frequently/recently accessed data. The gamble is: most of the user-driven data fetches and writes will require the same pages
- To ensure that there are enough free buffers, a dedicated process is responsible for writing dirty buffers to disk, which allows them to be re-used for different pages (_The lazywriter_ in SQL Server, _The background writer_ in PostgreSQL etc.)
- All databases rely on some variation of LRU (least recently used) algorithm to know when to evict a page from memory
- The page header is typically used to keep the information about the time the page was referenced, together with _is_dirty_ flag

## Write-Ahead Log (WAL)

- When user makes changes to the data and commits, the changes are not written to the data files immediately, this would be very inefficient (too often and non-sequential writes)
- Instead, the changes are made to the pages in the buffer pool, and those pages are marked as dirty (using a field on the page header). This is super-fast but not persistent
- In order to make those changes persistent, an entry related to a page modification is also immediately written to the **Write-Ahead Log (WAL)**
- All the WAL writes are sequential, which makes them very fast
- So most of the time during database operation, the data files don't contain the most up-to-date state (are inconsistent); however, this does not impact users who can only see the data in the buffer pool
- In case of a failure, the database reads through WAL and replays all the completed operations in order to restore the most recent committed state
- In order to avoid long restore times and ever-growing WAL, a dedicated **checkpoint process** runs periodically to write all the dirty pages back to the data files
- **Checkpoint** is a point up to which all the changes are guaranteed to be saved to disk
- Once done, the checkpoint process creates a WAL entry that indicates the checkpoint completion
- This means that, in the event of a crash, the crash recovery procedure have to find the latest checkpoint record to determine the point in the log from which it should start replay

## Table and index data structures

- The most basic data structure of a table is a **heap**: basically, a big area of memory where rows are stored unordered
- In order to find data in a heap, you need to perform the **full scan**, i.e. read all of the rows in a table and check the filtering condition for each row
- Typically, relational databases provide mechanisms that allow data to be physically stored in a specific order (Clustered index in SQL Server, Index Organized Tables in Oracle etc.)
- Physical ordering allows to optimize data access, but only when the filtering condition matches the sorting order; in order to support additional data access paths, additional **indexes** may be used
- The most common structure of an index is a **B-tree** (can be thought as a multi-level index)
- B-tree is a structure similar to a binary search tree, but which is optimized for how the database stores data on disk
- Basically, each one node of a b-tree fits into one block (a minimal unit to be read or written to disk), and thus may contain more than 2 children
- The leaf nodes contain index records, each index record is a combination of an index key and a pointer to a table row (in case clustered index, the leaf nodes contain the actual data)
- The non-leaf nodes contain pointers to the next level index nodes which have keys `k_i` such as `n < k_i < m` for all `i`
- Search is trivial, deletion and especially insertion are more complicated
- Upon insertion, if there is enough space in the block, the restructuring is limited to one block. However, if the block capacity is exceeded, then the block is split into two blocks, and the update is propagated to upper levels

## Indexing

- **Cardinality**: number of distinct values in a particular column that an index covers
- **Index selectivity:** cardinality / total number of records (a highly selective index filters out more rows in a search)
- **Compound indexes** use multiple columns
- Usually databases have the ability to combine multiple indexes (using bitmaps)
- **Covering index** includes all the fields used by the query, which allows **index-only scans**
- **Unique indexes** are used to enforce uniqueness of a column's value (different from unique constraint)
- **A partial/filtered index** is an index built over a subset of a table; it contains entries only for those table rows that satisfy the predicate
- Usually creating a primary key results in creating an index, and you need to decide whether you want that index to be clustered or simply unique

### Best practices:

- Consider indexing keys that appear frequently in WHERE clauses
- Create indexes on foreign key columns
- Use narrow indexes: as small a data type as you can (INTEGER vs VARCHAR)
- Choose index keys that have high selectivity
- As a rule of thumb, when indexing by multiple columns, put the most selective column first (but sorting and grouping can affect this)
- When the data is physically ordered on one of the keys, then place this key first in the composite index
- Avoid using frequently updated columns for clustered index
- Avoid wide clustered keys (since all nonclustered indexes hold the clustered keys as their row locator)
- Avoid nonclustered indexes for queries that retrieve a large number of rows

## Transaction isolation

- Transaction is the unit of activity that is atomic (all or nothing)
- Database transactions have 4 properties commonly referred as ACID: atomicity, consistency, isolation, and durability

### Acid

- **Atomic**: either all of a transaction happens or none of it happens. If transaction fails, every change is rolled back; the database is left unchanged
- **Consistency**: a transaction takes the database from one consistent state to the next. By the end of the transaction every constraint is guaranteed to hold, every index is guaranteed to be updated etc.
- **Isolation**: the effects of a transaction may not be visible to other transactions until the transaction has committed
- **Durability**: once the transaction is committed, it is permanent. The results are stored on non-volatile memory (disk)
- Out of all properties, isolation is typically the one that can be relaxed (in favor of concurrency)

### Read phenomena

- The ANSI/ISO standard SQL 92 describes three different read phenomena when a transaction retrieves data that another transaction might have updated
- **Dirty read**: read of uncommitted data from another concurrent transaction
- **Nonrepeatable read**: a transaction re-reads data it has previously read and finds that data has been modified by another transaction (that committed since the initial read)
- **Phantom read**: a transaction re-executes a query returning a set of rows that satisfy a search condition and finds that the set of rows satisfying the condition has changed due to another transaction (that committed after the initial query was executed)
- Transaction isolation levels are defined by the presence or absence of these phenomena

### Other consistency-related problems

- **Lost updates**: the uncommitted data written by a transaction is overwritten by another transaction before the first transaction commits
- **Serialization anomaly**: the result of successfully committing a group of transactions is inconsistent with all possible orderings of running those transactions one at a time

### Isolation levels

- **Read uncommitted**: statements can read rows that have been modified by other transactions but not yet committed. Any of phenomena can be observed
- **Read committed**: statements cannot read data that has been modified but not committed by other transactions. This prevents dirty reads. With some exceptions, this typically is a **default** isolation level
- **Repeatable read**: statements cannot read data that has been modified but not yet committed by other transactions and no other transactions can modify data that has been read by the current transaction until the current transaction completes. This prevents nonrepeatable reads and lost updates
- **Serializable**: emulates serial transaction execution for all committed transactions: transactions are guaranteed to produce the same effect as running them one at a time in some order. This prevents phantom reads and serialization anomalies

## Locks

- **Optimistic concurrency** makes the optimistic assumption that collisions between transactions will rarely occur
- **Pessimistic concurrency** makes the assumption that collisions are commonplace. Pessimistic concurrency relies on **locks** to control concurrent access to shared resources
- Granularity of locks: the level of an object in a hierarchy that the lock is applied to, e.g. row-level, table-level etc.
- When number of locks grows, databases may apply **lock escalation**, where fine-grained locks are replaced by a single coarse-grained lock
- **Shared mode** allows a resource to be locked by several processes at a time; used for reading
- **Exclusive locks** are incompatible with all the other types of locks and can only be held by one process at a time; used for writing
- Locking may result in a **deadlock**, when two transactions are blocking each others progress. Databases usually detect this situation and abort one of the transactions
- The main rule to avoid deadlocks is to acquire locks on multiple objects in a consistent order

## Query execution

- SQL is a declarative language: queries specify what data to fetch, but not how to fetch it
- Any query may have several execution paths, the job of the query optimizer is to build an optimal **execution plan**
- TODO: sequential table scan, index scan
- TODO: lookups
- Joins are executed using one of three methods: nested loop join, merge join, or hash join
- **Nested loop join**: the outer loop traverses all the rows of the first set; for each of these rows, the nested loop goes through the rows of the second set
- **Merge join** requires both inputs to be sorted based on the join column(s). The algorithm then scans both sides and merges the rows
- **Hash join** algorithm builds hash table on one of the inputs and then uses it to search for matching rows. A hash join is most efficient when the whole hash table fits in RAM, this is why the smaller input of the two is usually used as the input for building the hash table
