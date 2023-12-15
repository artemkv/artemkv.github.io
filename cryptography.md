# Cryptography

TL;DR
- SHA-3 for cryptographic hashing
- MD5 is fine for checksum and other non-cryptographic purposes
- PBKDF2 (Salt + repeated hashing) for storing password hashes (or when deriving a secret key from a passphrase)
- AES (+ 256bit CBC as a default choice) for symmetric encryption
- Diffie–Hellman if you need to securely communicate the shared key (for a single use) for symmetric encryption and you can only use the public channel
- RSA for asymmetric encryption
- Encrypt hash (digest) using private key to produce a digital signature
- Encrypt session keys using public key for a secure session


## Hashing

- Hashing allows to calculate a digest (a "fingerprint") of a large file
- Hash is typically much smaller than the original file
- Thus, comparing hashes is much faster than comparing the original files, so it can be used instead
- However, this can produce false positives because of hash collisions (2 different files having the same hash)
- **Cryptographic hash functions** have many applications in information security field (e.g. digital signatures)
- Hash function has to be deterministic
- Ideally, changing 1 bit of the input should change about half of the output bits
- Should have a high **preimage resistance** (function should be "one-way"): it should be computationally infeasible to find an input of a hash function knowing the output
- In other words, given `y`, you can't find any `x` such that `hash(x) == y`
- **2nd preimage resistance**: given `x`, `y`, such that `hash(x) == y`, you can't find `x'` where `x' != x` and `hash(x') == y`
- Should have a high **collision resistance**: it should be computationally infeasible to find two distinct messages that hash to the same value
- In other words, nobody can find any `x`, `z` such that `x != z` and `hash(x) == hash(z)`
- Merkle and Damgard in 1979 found a way to build a collision-resistant one-way hashes (with the mathematical proof and all)
- MD5 was the first implementation, released in 1992 (as a fun fact, "MD" stands "Message Digest", not "Merkle-Damgard"), produces 16 byte-long values
- MD5 was later discovered to have a weakness, massively reducing the order of magnitude required for a brute-force attack trying to find a collision (a different message having the same hash)
- Preimage resistance is typically much harder to break
- MD5 can still be used as a checksum and other non-cryptographic purposes
- SHA family of algorithms was ordered by the US government to replace MD5
- SHA-1 was deprecated in 2011
- SHA-2 is believed safe so far
- All Merkle-Damgard algorithm are susceptible to length extension attack
- Explanation: if I know the hash `H(m)` and the length of the original message `m`, I can calculate `H(m||z)`, where `z` is a sequence of bytes I want to append at the end of `m`, without knowing [the full] `m`
- In practice: imagine `m` is `s||d`, where `s` is a secret and `d` is data, and `H(s||d)` is calculated to "sign" the data part `d`. Being a man in the middle, I don't know `s`, but I see `d`, `H(s||d)` and I know the length of `s`. This allows me to construct `d||z`, and valid `H(s||d||z)`. When the receiver tries to validate `d||z` and `H(s||d||z)`, it will appear as valid
- You can double-hash to protect against the attack
- SHA-3 deviates from Merkle-Damgard algorithm, producing the hashes of the same length as SHA-2, but making it impossible to use the length extension attack
- You can use openssl to compute the file hash:

```
openssl dgst -sha512
```


### Storing passwords (not)

- It's a good idea to store password hashes instead of passwords in your DB and compare hashes instead of comparing passwords
- This would prevent the disaster if the copy of your DB leaks into the world
- If you just use the original password when calculating the hash, the same password will produce the same hash, so the hackers could spot the accounts with the same passwords in your DB
- The hacker could even pre-calculate the hashes of the most popular passwords (**rainbow table**), and use this table to find the accounts with those passwords
- You can avoid this by using **Salt**: a random piece of information mixed in together with the original password to produce hash. Salts are stored in clear, but each record has to use a different salt
- Still, hacker could brute-force trying different salts. To slow them down, hashing is typically repeated multiple times (typically tens of thousands times)
- PBKDF2 is a standard implementation and is recommended for password hashing


## Symmetric Encryption

- Based on shared secret
- The key is used both for encryption and decryption
- The input text is called **plain text**
- The encrypted text is called **cyphertext**
- The shared secred should be protected by limiting the audience with whom it is shared, sharing the key using different channel, re-generating the compromised keys, using the key only for short period of time and avoiding revealing the plain text (as the attacker could reverse-engineering the key from it)
- Ideally, the symmetric key should only be used once (in which case it is called **Session key**)
- Don't encrypt data from third parties, to avoid your key to be reverse-engineered. If you absolutely must do so, add some random prefix
- Naive symmetric key encryption can be broken by statistical analysis of the cyphertext. You could calculate a probability of a certain key to have been used to encrypt the plain text
- To combat statistical analysis, **confusion** and **diffusion** are used
- **Confusion** means that each bit of the ciphertext should depend on several parts of the key
- So, confusion hides the relationship between the ciphertext and the key
- **Diffusion** means that if we change a single bit of the plaintext, then (statistically) half of the bits in the ciphertext should change (and vice versa)
- So, diffusion hides the relationship between the ciphertext and the plain text
- First approaches to introduce confusion were developed by Horst Feistel in 1973 (IBM)
- DES used to be the most popular algorithm based on Feistel's research
- However, the DES used a very small key (56 bit), which made the brute-force attack quite cheap
- To combat this weakness, the workaround of applying DES 3 times was commonly used ("Triple DES")
- New "Advanced Encryption Standard" (AES) replases DES
- To ensure diffusion, ecryption is done in blocks, using the previous block to encrypt the next one
- Various modes of applying diffusion can be used with AES
- In ECB mode, each block is encrypted separately. This basically means "no diffusion"
- With ECB, the same plaintext block will result in the same ciphertext block (repeated characters of the plaintext will produce repeated characters of the ciphertext)
- In CBC mode, each block of plaintext is XOR-ed with the previous ciphertext block before being encrypted
- When in doubt, use CBC as a default choice
- You can generate the random key and encrypt-decrypt the file using openssl:

```
openssl rand -hex 32
openssl enc -aes-256-cbc -in xxx.bmp -K XXXXXXXXXXXX > xxx.enc
openssl enc -aes-256-cbc -in xxx.enc -K XXXXXXXXXXXX -d > xxx.bmp
```


### Sharing the key using passphrase

- Sharing the 256-bit key is tricky, one of the techniques is to use the passphrase that is easy to communicate over the phone and use the algorithm to derive the actual key from that passphrase
- The passphrase has to have enough entropy to avoid brute-force attack (simply using one-english-word password would have too little entropy and can be easily brute-forced)
- You should apply this technique using the same preventive methods as with storing passwords (using salt and multiple iterations of hashing)
- Same as with password hashes, you can use PBKDF2
- Encrypting-decrypting using openssl and pbkdf2 (with default number of iterations and generating random salt):

```
openssl enc -aes-256-cbc -in xxx.bmp -pbkdf2 > xxx.enc
openssl enc -aes-256-cbc -in xxx.enc -pbkdf2 -d > xxx.bmp
```

- The command will prompt for the passphrase
- Salt that is generated randomly will be put in the beginning of the encrypted file (prefixed with "Salted__")


## Asymmetric encryption

- Asymmetric encryption is a cryptographic system which uses pairs of keys: public keys (which may be known to others), and private keys (which may never be known by any except the owner)
- Diffie–Hellman key exchange and Public Key Infrastructure are both based on asymmetric encryption


## Sharing the key using Diffie–Hellman key exchange algorithm

- Diffie–Hellman algorithm solves the problem of communicating the cryptographic key for the symmetric encryption (and avoids using passphrase-derived keys)
- It allows securely exchanging cryptographic keys over a public channel
- Algorithm is based on a **trapdoor function**
- A trapdoor function is a function that is easy to compute in one direction (computing `f(a)` having `a`), yet difficult to compute in the opposite direction (finding its inverse, or computing `a` knowing `f(a)`)
- If Bob uses secret (a private key) `a`, he can produce `f(a)` (public key); if Alice uses secret `b`, she can produce `f(b)`. Now, by using a function `h` so that:

```
h(a, f(b)) == h(b, f(a))
```

- ...Bob and Alice can both calculate it using only their respective private keys and the other party's public keys. They can use the calculated value as a symmetric key for the further communication. (`f` has to be a trapdoor function, and also have the corresponding `h`)
- In this scenario, the attacker can clearly see `f(a)`, `f(b)` and knows how to calculate `f(x)`, but he cannot reverse-engineere `a` or `b`
- Function `f(a)` in Diffie–Hellman algorithm is `f(a) = g^a mod p`, where `g` and `p` are public and shared between two parties
- To make the key difficult to break, we need to make sure `f(a)` produces all possible values; for that `g` and `p` have to be properly selected (the exact rules are quite complicated, but in short, they are based on prime numbers)
- openssl can be used to generate a parameter file (containing `g` and `p`), and (from the parameter file) public and private keys:

```
openssl genpkey -genparam -algorithm DH -out params.pem
openssl genpkey -paramfile params.pem -out alice_private_key.pem
openssl pkey -in alice_private_key.pem -pubout -out alice_public_key.pem
```

- Parameter file then have to be shared with the other party (over an open channel), so that party can also use it to generate their private and public keys:

```
openssl genpkey -paramfile params.pem -out bob_private_key.pem
openssl pkey -in bob_private_key.pem -pubout -out bob_public_key.pem
```

- Public keys are exchanged (over an open channel)
- "pkeyutl -derive" can then be used to derive the shared secret for the symmetric encryption from the private key and other party's public key:

```
openssl pkeyutl -derive -inkey bob_private_key.pem -peerkey alice_public_key.pem -out shared_secret.bin
```

- Hash the derived shared secret to produce an actual AES key!
- Don't reuse keys!
- Be aware that this process is heavy so it become a potential target for DOS attack


## Public Key Infrastructure (PKI)

- With PKI, public key represents someone's identity
- To make it work, that person needs to have a private key and never share it with anyone
- RSA is the most popular algorithm for PKI
- It uses a different trapdoor function, based on the multiplication of 2 very large prime numbers. It is easy to multiple numbers, but it's very difficult to factorize the product into the pair of prime numbers

### RSA algorithm:

- Find 2 different large prime numbers `p` and `q`. This should be kept secret
- Calculate `n = pq`. `n` is the modulus for the public key and the private keys
- It is very easy to find `n` but is very difficult to find `p` and `q` from `n`
- Choose `e` having no common factor with `(p-1)(q-1)`. Easiest way to do it is to use a prime for `e`
- `e` doesn't have to be a secret, popular choise is 65537
- `e` is the public key exponent
- Find `d` to satisfy `ed-1 = h(p-1)(q-1)`
- This step can be done iteratively, trying different `h` and gradually moving towards the correct one (extended Euclidean algorithm)
- `d` is the private key exponent
- Public key: `n`, `e`
- Private key: `p`, `q`, `d`
- Private key function: `c = m^e mod n`
- Public key function: `m = c^d mod n`
- The algorithm depends on the fact that it's extremely difficult to find `d` knowing only `c` and `n`. You need to know original `p` and `q` that go into `n` to find `d`
- Basically, it's very difficult to find a pair of numbers `e` and `d` to create those 2 inverse functions
- All the "weird" steps in the algorithm are basically tricks to find a possible value for `d` from knowing `p`, `q` and `e` and in a reasonable time
- You can generate public and private keys for RSA using openssl:

```
openssl genpkey -algorithm RSA -out private_key.pem
```

- To analyze the generated private key:

```
openssl rsa -in private_key.pem -text -noout
```

- modulus is `n`, prime1 and prime2 are the `p` and `q`, public exponent is `e`, private exponent is `d`, coefficient is `h`
- exponent1 is `d(mod p − 1)` and exponent2 is `d(mod q − 1)`; these values allow to run the decryption operation faster than if you only had `d` and `n`
- To extract the public key from the private key:

```
openssl rsa -in private_key.pem -pubout -out public_key.pem
```

- To analyze the extracted public key:

```
openssl rsa -pubin -in public_key.pem -text -noout
```

- modulus is `n`, exponent is `e`


### Digital signatures

- Apply hash function (SHA-256) to the original document to produce digest, the encrypt the digest using the private key. The encrypted digest is the signature
- This can be done with openssl:

```
openssl dgst -sha256 -sign private_key.pem -out signature.bin document.doc
```

- Share the original document (document.doc) and the signature (signature.bin) with someone who has the corresponding public key
- To verify the signature, apply hash function (SHA-256) to the received document to produce digest yourself, decrypt the signature using the public key to produce the digest computed by the sender, and compare two values
- This can be done with openssl:

```
openssl dgst -sha256 -verify public_key.pem -signature signature.bin document.doc
```

- As the private key is only known the sender, and the message can only be decrypted by the corresponding pubic key, only the sender could create the correct signature
- This not only confirms the document is coming from the correct source, but also that it wasn't modified in transit
- Only sign the document you yourself issue!


### Lamport signature

- Lamport signature a method for constructing a digital signature from a hash function (or any cryptographically secure one-way function)
- Step 1: generate private key as purely random number, organize it logically into 2 rows of 256 blocks each, with each block being 32 bytes long
- Step 2: for each of these blocks, take a hash. Re-arrange the hashes in the same way as original blocks, this is your public key
- Step 3: Signing a message. Take hash of a message, 256 bits long. Pick the blocks of a private key to reveal, based on bits in the message hash: if the bit is zero, reveal the block from the first row, if the bit is one, reveal the block from the second row. This sequence of revealed blocks from private key is your signature
- Distribute your public key, message and the signature
- Step 4: Verifying the message. Hash each block of the signature and match with a corresponding block of a public key
- This also reveals the half of the private key, but this is not a great problem, as this would only allow you to sign the exact same message (technically, the message with the same hash)
- This algorithm would still be secure in the age of quantum computers
- The disadvantage is that each private key can only be used once
- Also, it is quite big (8kb for signature)
- There are optimizations you can do to mitigate both problems
- One, you can use 32 bytes long private key, and generate infinite number of public key blocks by hashing the private key with incrementing suffix (a counter): `h0 = hash(pk + 0)`, `h1 = hash(pk + 1)` etc.
- Every time you double the public key, you double the number of messages you can sign with it
- Second, instead of revealing the whole public key (speaking before signing), you can reveal only the hash of public key, and thus "commit" to your public key. Of course, you would need to send the whole public key together with the signature to be able to verify the hash
- This last problem can be mitigated by committing to a merkle tree of hashes, this way to verify the public key, instead of sending the whole key, you can send the small part of the key, and hashes for the remainder of the key
- Merkle trees were invented specifically for lamport signatures (around 1976)


### Session keys

- If two parties want to exchange the data securely, they need a common secret, which needs to be rotated often
- The common practice is to use the **Session Key** - the key for the symmetric encryption that is only valid for a duration of a session
- To share that key, Alice can encrypt it using Bob's public key, so that only Bob can decrypt it using his private key


### Digital Certificates

- x.509 is a standard
- **Subject** is the web site or an app that a certificate protects (e.g. example.com)
- **Issuer** is an authority that issued the certificate (e.g. "CertSign CA")
- **Validity** denotes the time during which the certificate is valid
- **Public Key** is the one of a subject
- **Signature** is created by the issuer, can be validated using the issuer's public key
- Subject and the issuer are specified using **Distinguished names**
- **Distinguished name** consists of: Country, State, Locality, Organization, Organizational Unit (optional), Common Name
- In case of website, Common Name is the domain name
- To prove the validity of a website certificate, you need an issuer's public key, which you can get from the issuer's certificate
- But before you use the issuer's certificate, you have to validate it too
- So you basically have to follow the chain of the certificates up to the **Root Certificate**
- The Root Certificate is signed by itself
- There is a very small number of root certificates and they come preinstalled with your OS by the OS vendor
- In the end, you have to trust someone
- You can create a request for a new certificate using openssl:

```
openssl req -new -key private_key.pem -days 60 -out csr.pem
```

- To analyze the request:

```
openssl req -in csr.pem -text -noout
```

- Then you send the request to a certificate authority and receive a certificate
- Once you receive the certificate, you can use openssl to analyze it:

```
openssl x509 -in certificate.cer -text -noouts
```

- To install the certificate to the web server (e.g. IIS), you need to combine the received certificate with your private key
- You can do it using openssl:

```
openssl pkcs12 -export -in certificate.cer -inkey private_key.pem -out combined.pfx
```

- The combined file can be installed in IIS, for example
- This process is not done manually anymore. These days the process is automated and happens behind the scenes. You are supposed to configure the certbot that automatically renews the certificates for you
- In practice, SSL is often offloaded on the LB level