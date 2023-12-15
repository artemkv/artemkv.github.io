# TCP/IP

## Key Performance Characteristics

- **Bandwidth** - how much data can be sent per unit of time
- **Latency** - how long it takes for a request to arrive

```
8   Mbps == 1    MB/s
100 Mbps == 12.5 MB/s
```


## Operation Modes

- **Simplex Operation:** can send data in only one direction
- **Half-Duplex Operation:** can send data in both directions, but only one direction at a time
- **Full-Duplex Operation:** can send data in both directions simultaneously


## OSI Layers (legacy)

- **Physical Layer:** topology, cables, transform data from bits into signals, signal transmission
- **Data Link Layer:** put data into frames, managing the medium
to avoid conflicts, addressing (MAC address), error detection (CRC)
- **Network Layer:** interconnected networks, logical addressing (IP address), routing, fragmentation and reassembly, error handling
- **Transport Layer:** communication between processes, process-level addressing (TCP/UDP ports), multiplexing and demultiplexing data from multiple processes, acknowledgments
and retransmissions
- **Session Layer:** setting up, managing, and ending sessions
- **Presentation Layer:** char sets, byte order, compression, encryption
- **Application Layer:** application-specific messages


## TCP/IP Layers

- **Physical Layer:** N/A
- **Data Link Layer:** normally handled by the local area network (LAN) technology (Ethernet, Token Ring etc), but can be handled by SLIP or PPP
- **Network Layer:** IP
- **Transport Layer:** TCP/UDP
- **Session Layer, Presentation Layer, Application Layer:** combined: HTTP, FTP, SMTP etc.

```
LAN (Ethernet):  Frame
IP:              Datagram
TCP:             Segment
UDP:             Packet
HTTP:            Message
```

### Network devices

- **Hub** connects multiple devices on one network and makes them act together as a single network. A hub has multiple input/output ports, in which an input in one port results in it being an output in all the other ports, except the port where it was input. The receiving port that has to decide if the data packet is actually intended for it by checking the address on the packet, before passing it on further.
- A hub operates at the Physical layer.
- **Switch** connects multiple devices on one network and makes them act together as a single network. Switch gathers information about the data packets it receives (Which MAC address is connected to which port) and forwards it to only the network that it was intended for. When a switch receives a data packet, it examines the data address, the sender and the receiver and sends the data to the device that the data is meant for. Switch uses hardware addresses to filter the network.
- A switch operates at the Data Link layer.
- **Router** perform packet switching, filtering, and path selection. Router uses logical addressing to filter the network.
- A router operates at the Network layer.

```
Hubs have one collision domain.
Switches create separate collision domains within a single broadcast domain.
Routers provide a separate broadcast domain for each interface.
```



## SLIP/PPP

- Used when there is no underlying local area network technology
- **SLIP** takes the IP datagram, send it one byte at a time, and then send the byte 192 to delimit the end of the datagram
- **PPP** is more advanced, provides:
    - frame size negotiation
    - multiplexing
    - error detection (CRC)
    - quality control
    - etc.


## ARP (TCP/IP Address Resolution Protocol)

- **Address resolution:** MAC address <-> IP Address
- **Direct Mapping**: incorporate MAC address as a part of IP Address. Could work if IP address was big enough, but is not done that way
- **Dynamic Address Resolution**: use the protocol to find out mappings (think "Limousine driver waiting to pick up a person at the airport")
- Alice wants to send message to Bob (129.168.0.5), but she does not know Bob's address (MAC Address)
- She broadcasts message "Who is Bob?" on level 2 (Data Link Layer)
- Only Bob replies, sending his address (MAC Address)
- Improvements:
    - **Caching:** store the reply and don't ask again (expires automatically in 10-20 minutes)
    - **Cross-resolution:** send your own address when asking for the someone else's address


## IP

- Goal: the delivery of datagrams across an internetwork of connected networks
- Key characteristics:
    - Universally Addressed
    - Underlying Protocol-Independent
    - Connectionless Delivery
    - Unreliable Delivery (best-effort delivery)
    - Unacknowledged Delivery
- Functions:
    - **Addressing** independent of the underlying network protocols
    - **Data Encapsulation** into an IP datagram
    - **Fragmentation** and **Reassembly** to make sure the datagrams do not exceed underlying network protocol limits
    - **Routing** datagrams to the next device


### IP Addressing

#### Classful Addressing (legacy)

- Original scheme. Class defines number of bit for a network and number of bits for a host. Innefficient, inflexible, huge routing tables.
- **Class A:** Very large organizations with hundreds of thousands or millions of hosts

    `0### #### **** **** **** **** **** ****`

- **Class B:** Medium to large organizations with many hundreds to thousands of hosts

    `10## #### #### #### **** **** **** ****`

- **Class C:** Small organizations with no more than about 250 hosts

    `110# #### #### #### #### #### **** ****`

- **Class D:** For multicasting

    `1110 MMMM MMMM MMMM MMMM MMMM MMMM MMMM`

- **Class E:** For experimental use

    `1111 EEEE EEEE EEEE EEEE EEEE EEEE EEEE`

```
127.0.0.1 == localhost
```

#### Subnetted Classful Addressing (legacy)

- Same as in classful scheme, class defines number of bit for a network. The rest of bits are used for a **subnet** and a host. The number of bits used for subnet can only be known given a subnet mask.

```
Example for Class B address:

IP:             10## #### #### #### **** **** **** ****
Subnet mask:    1111 1111 1111 1111 1111 1111 1000 0000
                <     network     > <  subnet >< host >
```

- Subnets are invisible for the external world
- Subnets are undetectable without a subnet mask
- You can subnet multiple times creating hierarchy of subnets
- Organisations still have to buy IP addresses in blocks bound to classes

#### Classless Addressing

- Eliminate the prior notion of classes, and instead always use subnet masking. Subnetting can be done multiple times creating multi-level hierarchy of subnets


### Routing

- **Direct Datagram Delivery:** on the same physical network. Uses ARP to know MAC Address of the recipient.
- **Indirect Datagram Delivery**, also called **routing**, on the different physical network. Sends data to the router (the IP address of the datagram will still be that of the ultimate destination). Uses ARP to know MAC Address of the router. Final step is always a direct delivery.
- Sender must determine whether the datagram can be delivered directly or if routing is required, based on IP addressing
- **Classful Addressing** (legacy): We know the class of each address by looking at the first few bits. This tells us which bits of an address are the network ID. If the network ID of the destination is the same as our own, the recipient is on the same network; otherwise, it is not.
- **Subnetted Classful Addressing** (legacy): We use our subnet mask to determine our network ID and subnet ID and that of the destination address. If the network ID and subnet are the same, the recipient is on the same subnet. If only the network ID is the same, the recipient is on a different subnet of the same network. If the network ID is different, the destination is on a different network entirely.
- **Classless Addressing:** We use the slash number to determine what part of the address is the network ID and compare the source and destination. Used since around 1990.

#### Next-Hop Routing

```
Routing is done step by step, one hop at a time
```

- Routing tables are built automatically by multiple algorithms: RIP, OSPF, BGP etc.
- When routing, we take the destination IP, AND with Netmask and try to match with the Destination.
- If the match is found, Gateway and Interface define what happens next.
- Example:

```
Destination          Netmask            Gateway            Interface
0.0.0.0              0.0.0.0            10.10.11.1         10.10.11.51
10.10.11.51          255.255.255.255    127.0.0.1          127.0.0.1
10.255.255.255       255.255.255.255    10.10.11.51        10.10.11.51
127.0.0.0            255.0.0.0          127.0.0.1          127.0.0.1
224.0.0.0            240.0.0.0          10.10.11.51        10.10.11.51
255.255.255.255      255.255.255.255    127.0.0.1          127.0.0.1
```

- Our IP address is 10.10.11.51
- 127.0.0.1 = localhost
- Netmask 0.0.0.0 will zero all bits, making any IP address match 0.0.0.0. This is the default route. The default route says you have to send the datagram to 10.10.11.1, the default router. Use ARP to find out the MAC address and use Ethernet data link layer to send the datagram.
- Netmask 255.255.255.255 will keep all the bits, forcing the exact match on the complete IP address. When matched with 10.10.11.51 (our own IP address), the route is to send it to localhost (itself).
- Netmask 255.255.255.255 will keep all the bits, forcing the exact match on the complete IP address. When matched with 10.255.255.255 (broadcast address), use local network to do the broadcast.
- Netmask 255.0.0.0 zeroes all except the first byte of the IP address. Any address with 127 in the first byte will match 127.0.0.0, forcing it to be routed to localhost (itself).
- Netmask 255.0.0.0 zeroes all except the first 4 bits of the IP address. Any address with 0xE (224) in the first 4 bits will match 224.0.0.0 (multicast address), in that case use local network to do the multicast.
- Netmask 255.255.255.255 will keep all the bits, in case of exact match 255.255.255.255 (broadcast address), will route to localhost (itself).

### IP Datagram Format

- Divided into two pieces: the **header** and the **payload**
- The header contains:
    - *Source Address*, like 192.168.0.7
    - *Destination Address*, like 192.168.0.5
    - *Header Checksum*
    - *Protocol:* simply put, TCP or UDP? (can also be other like ICMP, but not something high level like HTTP)
    - *TTL (time to live):* specifies how long the datagram is allowed to live on the network, in router hops. Each router decrements the value of the TTL field (reduces it by one) prior to transmitting it. If the TTL field drops to zero, the datagram is assumed to have taken too long a route and is discarded.

#### IP Datagram Fragmentation

- The data link layer implementation puts the entire IP datagram into the data portion (the payload) of its frame format. If the datagram is bigger than the maximum frame size of the underlying network, it is necessary to fragment the datagram.
- Each device on an IP internetwork must know the capacity of its immediate data link layer connection to other devices. This capacity is called the **maximum transmission unit (MTU)** of the network, also known as the **maximum transfer unit**. Routers are required to handle an MTU of **at least 576 bytes**.
- Ideally, we want to use the largest MTU possible without requiring fragmentation for its transmission. To determine the optimal MTU to use for a route between two devices, we would need to know the MTU of every link on that route. One way to discover the best MTU for the route is to send test datagrams of various sizes with "Don't Fragment (DF)" flag. If the datagram is too big for one of the links, it must be discarded and a "Destination Unreachable" ICMP message must be sent back to the source.
- Fragmenting works by splitting payload and applying new (adjusted) headers based on the original header: `HHHHDDDDDDDD => H'H'H'H'DDDDDD + H''H''H''H''DD`
- Challenges:
    - *Re-assemble datagrams in a correct order.* Use "Fragment Offset" flag.
    - *Re-assemble datagrams that we fragmented more than once.* Use "Fragment Offset" flag.
    - *How to know this was the last diagram?* Use "More Fragments" flag.

```
Intermediate devices do not perform reassembly; reassembly happens only at the message’s ultimate destination.
```

- Motivation:
    - Fragments can take different routes
    - Simplicity
    - Reassembly of a message requires that we wait for all fragments before sending on the reassembled message

- Reassembly:
    - Check "More Fragments" flag to know if datagram was fragmented
    - Initialize the buffer
    - Start timer to know when to time out
    - Write data in the buffer starting from "Fragment Offset" flag value
    - Keep track of the portions of the buffer that have been filled


### NAT (IP Network Address Translation)

- Main Advantages
    - Public IP Address Sharing
    - Increased Security

#### Example

- Given:
    - Inside network that uses private addresses from the 10.0.0.0/8 address range
    - 20 inside global addresses from 194.54.21.1 through 194.54.21.20
    - Device 10.0.0.207 wants to access the server at public address 204.51.16.12
- Original Datagram:
    - Source Address: 10.0.0.207
    - Destination Address: 204.51.16.12
- After NAT Router
    - Source Address: 194.54.21.11
    - Destination Address: 204.51.16.12
- Server Response
    - Source Address: 204.51.16.12
    - Destination Address: 194.54.21.11
- After NAT Router
    - Source Address: 204.51.16.12
    - Destination Address: 10.0.0.207
- Can be static (always map 10.0.0.207 to 194.54.21.11) or dynamic (choose the free IP address from the pool 194.54.21.1 - 194.54.21.20)
- Can be unidirectional (private clients making requests to publicly globally available server) or bidirectional (clients making request to private server, would require static mapping)

#### Port-Based NAT

- IP addresses can be futher shared using ports (TCP/UDP).
- IP Addresses 10.0.0.207 and 10.0.0.208 can be mapped to the same IP address 194.54.21.11, using different ports, e.g. 10.0.0.207:7000 maps to 194.54.21.11:7224 and 10.0.0.208:7000 maps to 194.54.21.11:7225
- Original Datagram
    - Source Address: 10.0.0.207 : 7000
    - Destination Address: 204.51.16.12 : 80
- After NAT Router
    - Source Address: 194.54.21.11 : 7224
    - Destination Address: 204.51.16.12 : 80
- Server Response
    - Source Address: 204.51.16.12 : 80
    - Destination Address: 194.54.21.11 : 7224
- After NAT Router
    - Source Address: 204.51.16.12 : 80
    - Destination Address: 10.0.0.207 : 7000

### IPsec

- Allows 2 devices:
    - Agree on a set of security protocols to use
    - Decide on a specific encryption algorithm for encoding data
    - Exchange keys

```
IPsec works at the network layer (IP)
SSL works above TCP/IP and below HTTP
```

- Original IP Datagram:
```
[IP Header][XXXXXXXXXXXXX]
```

- Transport Mode:
```
[IP Header][[IPsec Header][XXXXXXXXXXXXX]]
           <-        IP Payload         ->
```

- Tunnel Mode:
```
[New IP Header][[IPsec Header][[IP Header][XXXXXXXXXXXXX]]]
                              <-    Original Datagram    ->
               <-              New IP Payload            ->
```


### Mobile IP (legacy)

- Handles relatively infrequent mobility

```
Mobile IP was designed under the specific assumption that the attachment point would not change more than once per second
```

- Mobile IP accomplishes these goals by implementing a forwarding system for mobile devices. When a mobile unit is on its home network, it functions normally. When it moves to a different network, datagrams are forwarded from its home network to its new location.
- Think "Mail forwarding"
- You leave London for Tokyo for a couple of months. You tell the London post office (PO) that you will be in Tokyo.
- They intercept mail headed for your normal London address, relabel it, and forward it to Tokyo.


### ICMP (Internet Control Message Protocol)

- Errors:
    - *Destination Unreachable Message* - could not deliver
    - *Source Quench Message* - too many datagrams too fast
    - *Time Exceeded Messages* - datagram expired (TTL went down to zero)
    - *Redirect Message* - you can use another router as a default router
    - *Parameter Problem Messages* - IP datagram header parameter has incorrect value

- Original IP Datagram:
```
[IP Header][XXXXXXXXXXXXX]
```

- New IP Datagram:
```
[New IP Header][[ICMP Common Part][IP Header][XX]]
               <-        New IP Payload         ->
```
New payload contains just first N bytes from original payload.

- Info:
    - *Echo and Echo Reply Message* - ping
    - *Traceroute Message* - traceroute. Each router that sees that option while the test message is conducted along the route responds back to the original source with an ICMP Traceroute message.
- You could send multiple messages pogressivly increasing TTL from 1 to N, but that is not reliable


### TCP (Transmission Control Protocol)

- Key Characteristics:
    - Connection-oriented
    - Stream-based
    - Reliable delivery (data is retransmitted automatically)
- TCP is not responsible for maintaining message boundaries!
- TCP sends data as a continuous stream rather than discrete messages. It is up to the application to specify where one message ends and the next begins.

#### TCP Ports

- When an IP datagram is received, the Protocol field in the header indicates whether the datagram is destined for TCP or UDP protocol. Many processes on client machine with the same IP address send and receive data using TCP/UDP. To know to which process pass the data, we use ports.
- In UDP and TCP messages two addressing fields appear: a **source port** and a **destination port**. They identify the originating process on the source machine and the destination process on the destination machine.

```
TCP/IP transport layer addressing is accomplished by using TCP and UDP ports. Each port number within a particular IP device identifies a particular software process.
```

- Some ports are reserved for certain purposes (like 80 for web server). Universal agreement on port assignments is essential. It is possible for a particular web server to use a different port number, but in this case, the web server must inform the user of this number somehow, and user must explicitly tell the web browser to use it instead of the default port number.
- Each client process is assigned a temporary port number for its use. This is commonly called an **ephemeral port number**.
- The combination of IP address and the port number is called a **socket**.
TCP/IP application program interface (API) is called **sockets** (Windows Sockets or Winsock for Windows).

#### TCP Segments

- Upper layer protocols like HTTP just "pump" the data into TCP.
- For TCP the data that it receives is just a stream of bytes. It does not know where every HTTP message begins or ends.
- But it has to split the stream into descrete messages to pass to IP. These messages are called TCP Segments.
- TCP tries to pick the segment size so to avoid unnecessary fragmentation at the IP layer.
- A parameter called the **maximum segment size (MSS)** governs this size limit. Devices communicate MSS when establishing a connection.
- So one segment can contain pieces from 2 different HTTP messages.
- The HTTP messages must have some sort of explicit markers so that the receiving device can tell where one message ends and the next starts (two-character end-of-line sequence).
- As upper layer protocols just pump the data, TCP accumulates the bytes and decides when the amount of data is enough to create and send a segment. When an application has data that it needs to have sent across the internetwork immediately, even if the size is not yet enough for TCP, it can use the TCP **push** function.
- When this function is invoked, TCP will create a segment (or segments) that contains all the data it has outstanding and then transmit it with the **PSH** control bit set to 1. The destination device’s TCP software, seeing this bit sent, will know that it should not just take the data in the segment it received and buffer it, but rather push it through directly to the application.
- Push function only forces immediate delivery of data. It does not change the fact that TCP provides no boundaries between data elements. It is possible that the first push may contain data given to TCP earlier that wasn’t yet transmitted, and it’s also possible that two records pushed in this manner may end up in the same segment anyway.

#### TCP Connection Handshake

1. Client sends **SYN** to Server
2. Server replies with **ACK + SYN**
3. Client replies with **ACK**
- SYN contains Initial sequence number (ISN).
- In the normal case, each side terminates its end of the connection by sending a special message with the **FIN** (finish) bit set. The device receiving the FIN responds with an acknowledgment to the FIN that indicates that it received the acknowledgment. Neither side considers the connection terminated until they both have sent a FIN and received an ACK, thereby finishing the shutdown procedure.

#### TCP Message (Segment) Format

- Most important fields:
    - *Source Port*
    - *Destination Port*
    - *Sequence Number:* number of the first byte of data in this segment or ISN in case of SYN
    - *Acknowledgment Number:* the sequence number the source is next expecting the destination to send
    - *Window:* urrent size of the buffer allocated to accept data for this connection (size of the Send Window for the sender)
    - *Checksum:* computed over the entire TCP datagram, plus a special pseudo header of fields.
    - *Data:* payload

#### TCP Sliding Window

- TCP functions:
    - **Reliability** - ensuring that data that is sent actually arrives at its destination, and if it doesn’t arrive, detecting this (timeout) and resending it.
    - **Data Flow Control** - managing the rate at which data is sent so that it does not overwhelm the device that is receiving it.
- Bytes in the stream is logically divided into 4 categories:
    - Sent and Acknowledged
    - Sent but Not Yet Acknowledged
    - Not Yet Sent for Which Recipient Is Ready
    - Not Yet Sent for Which Recipient Is Not Ready

```
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
<- sent, -><- sent, -><-     not sent,    -><- not sent, ->
   acked    not acked        recepient         recepient
                             is ready          not ready
           <-         Send Window         ->               
```

- The receiver communicates the Window size when sending and acknowledgment.
Each time an acknowledgment is received, the Send Window slides across the stream.
- The left boundary is the first byte to be sent. The right boundary is defined by the windows size communicated by the receiver.

#### Silly Window Syndrome

- If the client is not able to cope with the incoming data, the Send Window shrinks. As a result, it can shrink to such a small size as one byte. The sender will end up sending a lot of very small messages.
- Solutions:
    - Do not reduce the window size, close the window completely, and reopen once enough space is available
    - Do not send as much data as possible, wait until we are able to create a segment of a reasonable size

### UDP (User Datagram Protocol)

- Key Characteristics:
    - Connectionless
    - Message-based
    - Best-effort delivery, no retransmissions
- UDP’s only real goal is to serve as an interface between networking application processes that are running at the higher layers, and the internetworking capabilities of IP.
- UPD takes a data, adds a header with source port, destination port, length and a checksum (on message + pseudo header) and passes it to IP.
- Usages:
    - Performance is more important than completeness
    - Data exchanges that are short and sweet


### DNS

- google.com = 172.217.17.110
- Can be used for load balancing.
- DNS Record types:
    - *A* - contains a 32-bit IP address for a host
    - *AAAA* - contains IPv6 address for a host
    - *CNAME* - contains a domain name alias. A CNAME-record should always point to an A-record and never to itself or another CNAME-record to avoid circular references
    - *NS* - specifies the name of a DNS name server that is authoritative for the zone

#### Resolution:

- Your local DNS server IP address is a part of your IP configuration. It actually doesn't store the DNS database records (except in cache), and only knows how to query actual DNS servers with records. So it is a DNS resolver.
- When you try to resolve example.com, you send the request to your local DNS server (DNS resolver).
- The local DNS server will send the request to 1 out of 13 root DNS servers.
- The root server will respond saying that you should ask another DNS server that is responsible for all .com domains (TLD Server). This is what NS record is used for: the root server will have an NS record for .com pointing to ns.server-for-com.net. It will also have an A record to tell you where that ns.server-for-com.net is.
- TLD Server responsible for .com domains will respond with the address of the authoritative name server responsible for example.com domain
- So, in essence, you repeat the question until you reach the DNS server that has an A record for example.com. That server is called authoritative name server.


### DHCP

- Rather than using a static table that absolutely maps hardware addresses to IP addresses, a pool of IP addresses is used to dynamically allocate addresses. Dynamic addressing allows IP addresses to be efficiently allocated, and even shared among devices.
- At the same time, DHCP still supports static mapping of addresses for devices where this is needed.
- With dynamic allocation, DHCP assigns an IP address from a pool of addresses for a limited period of time chosen by the server, or until the client tells the DHCP server that it no longer needs the address. An administrator sets up the pool (usually a range or set of ranges) of IP addresses that are available for use. Each client that is configured to use DHCP contacts the server when it needs an IP address. The server keeps track of which IP addresses are already assigned, and it leases one of the free addresses from the pool to the client. The server decides the amount of time that the lease will last. When the time expires, the client must either request permission to keep using the address (renew the lease) or must get a new one.


### Tools

- **hostname:** Prints the name of the current host.
- **ping:** Verifies that basic communication is possible between the TCP/IP software stacks on the two machines.
- **traceroute (tracert):** Shows the path taken by the IP diagram on the way to be delivered to the target address.
- **arp:** Displays and modifies the IP-to-Physical address translation tables used by address resolution protocol (ARP).
- **nslookup:** Translates name into IP address or vice versa
- **netstat:** Displays protocol statistics and current TCP/IP network connections.
- **ifconfig (ipconfig):** View and modify the software settings that control how TCP/IP functions on a host.