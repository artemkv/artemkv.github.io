# HTTP
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [HTTP/2 : What you need to know - Robert Boedigheimer](https://www.youtube.com/watch?v=krEhLbAOalE)
- [Explaining QUIC: the protocol that is both very similar to and very different from TCP By Peter Door](https://www.youtube.com/watch?v=sULCOKfc87Y)


## Some gotchas

- Improve bandwidth, avoid latency
- Increasing bandwidth has limits: page load improvement plateaus after 5Mbps
- Reducing the latency leads to linear improvement in page load time

## HTTP messages

- Request

```
GET /some/resource.txt HTTP/1.1
Accept: application/json
Host: www.example.com
```

- Response

```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 123

{
  "some": "value"
}
```

### Status codes

- 1XX Informational

```
101 Switching Protocols
```

- 2XX Successful

```
200 OK
201 Created (The entity body of the response should contain the various URLs for referencing the created resource, with the Location header containing the most specific reference)
202 Accepted
```

- 3XX Redirection

```
301 Moved Permanently (The response should contain in the Location header the URL where the resource now resides)
302 Found (moved temporarily, future requests should use the old URL)
303 See Other
304 Not Modified
307 Temporary Redirect (replaces 302 in HTTP/1.1)
```

- 4XX Client error

```
400 Bad Request
401 Unauthorized
404 Not Found
405 Method Not Allowed
408 Request Timeout
409 Conflict
```

- 5XX Server error

```
500 Internal Server Error
503 Service Unavailable
```

- Reason phrase is for humans only

### Methods

- To be compliant with HTTP Version 1.1, a server need implement only the `GET` and `HEAD`
- `HEAD` behaves exactly like the `GET`, but the server returns only the headers in the response (exactly those that a `GET` request would return)
- `GET` and `HEAD` are supposed to be safe methods
- `PUT` creates a new document or replaces it
- `POST` was designed to send input data to the server
- `TRACE` for diagnostics, returns request message it received in the body of its response
- `OPTIONS`: list of supported methods
- `DELETE` request the resource deletion (server is allowed to override the request without telling the client)
- There exist extensions that define extension methods (e.g. `LOCK`, `MOVE` etc.)

### Headers

- Request headers: `Host`, `Referer`, `User-Agent`, `Accept` (`Accept-*`), `If-*`, `Authorization`, `Cookie` etc.
- Response headers: `Vary`, `Set-Cookie` etc.
- Entity headers: `Location`, `Content-Length`, `Content-Type` (`Content-*`), `ETag`, `Expires`, `Last-Modified` etc.
- `Content-Length` is essential for persistent connections!!!


## Upgrading protocol

- Upgrade header is used to upgrade a connection to a different protocol
- Connection header allows a connection to stays open
- Example is upgrading to a WebSocket connection:

```
Connection: Upgrade
Upgrade: websocket
```


## Connection management (HTTP/1.0 -> HTTP/1.1)

- HTTP is layered directly on TCP
- **SYN/SYN+ACK handshake** creates a measurable delay
- Small HTTP transactions may spend 50% or more of their time doing TCP setup
- **TCP delayed acknowledgments** hold outgoing acknowledgments in a buffer for a certain window of time (usually 100–200 milliseconds), looking for an outgoing data packet on which to piggyback
- Frequently, the disabled acknowledgment algorithms introduce significant delays
- Sender has to wait for an acknowledgment to have a permission to send more packets
- **TCP slow start** throttles the number of packets a TCP endpoint can have in flight at any time
- TCP slow start is used to prevent sudden overloading and congestion of the Internet
- **Nagle's algorithm** discourages the sending of segments that are not full-size, attempts to bundle up a large amount of TCP data before sending a packet
- Small HTTP messages may not fill a packet, so they may be delayed waiting for additional data that will never arrive
- When a TCP endpoint closes a TCP connection, it maintains in memory a small control block recording the IP addresses and port numbers of the recently closed connection; this prevents any stray duplicate packets from the previous connection from accidentally being injected into a new connection that has the same addresses and port numbers
- This can lead to port exhaustion problems in benchmarking situations

### Optimizing HTTP connection performance

- Scenario: a web page with three embedded images
- **Serial connections:** retrieve all the resources one after another
- **Parallel connections:** open multiple connections and perform multiple HTTP transactions in parallel (retrieve 3 images in parallel)
- Allows overlapping delays, but consumes more bandwidth and multiplies number of parallel requests on a server; limited application
- **Persistent connections:** HTTP/1.1 allows HTTP devices to keep TCP connections open after transactions complete and to reuse the preexisting connections for future HTTP requests
- Nonpersistent connections are closed after each transaction. Persistent connections stay open across transactions, until either the client or the server decides to close them
- Allows to avoid handshake and slow start delays; but you may end up accumulating a large number of idle connections
- Today, many web applications open a small number of parallel connections, each persistent
- **Keep-alive connections:** early, experimental type of persistent connections (HTTP/1.0), deprecated in HTTP/1.1
- Clients implementing HTTP/1.0 keep-alive connections can request that a connection be kept open by including the `Connection: Keep-Alive` request header
- If the server is willing to keep the connection open for the next request, it will respond with the same header in the response
- Keep-alive does not happen by default, and `Connection: Keep-Alive` header must be sent with all messages that want to continue the persistence
- HTTP/1.1 assumes all connections are persistent unless otherwise
indicated
- HTTP/1.1 applications have to explicitly add a `Connection: close` header to a message to indicate that a connection should close after the transaction is complete
- An HTTP/1.1 client assumes an HTTP/1.1 connection will remain open after a response, unless the response contains a Connection: close
- The connection can be kept persistent only if all messages on the connection have a correct, self-defined message length (correct `Content-Length` or be encoded with the chunked transfer encoding)
- HTTP/1.1 permits optional **request pipelining** over persistent connections: multiple requests can be enqueued before the responses arrive
- There are limitations: responses must be returned in the same order as the requests, clients should not pipeline requests that have side effects etc.
- HTTP pipelining is not activated by default in modern browsers, as there are many potential issues (the implementations were extremely buggy)
- For example, pipelining is subject to the "Head-of-line blocking" problem: slow responses block all other requests and responses on that connection
- So in fact, no one is really using this
- For this reason, pipelining has been superseded by a multiplexing in HTTP/2


## HTTP/2

- HTTP/1.1 wasn't designed for modern web (average web page size is 2.5MB, making on average 100 requests)
- To optimize resource loading, most browsers use up to ~6 connections per host, all going through the handshake
- All the rest of the requests have to wait for one of those 6 currently pending requests to complete
- Persistent connections and pipelining attempt to solve it, but come with their own set of issues (see below) and no one is using pipelining
- There is no way to specify which requests are more important to get them through sooner
- While you can compress body, headers are not compressed (and have to be sent with every single request)
- HTTP/2 started as an experiment at Google (called SPDY, worked on top of HTTPS)
- Goals: single connection per host, header compression, request prioritization, server push
- HTTP/2 is standardized in 2015
- HTTP/2 is a binary protocol, no longer human-readable (you cannot simply telnet)
- HTTP/2 divides HTTP/1.x messages into frames which are embedded in a stream
- Several streams are combined together (multiplexing), to allow parallel requests to be made over the same single connection
- Data and header frames are separated, which allows header compression (using HPACK)
- HPACK allows to avoid sending the same headers again and again, by keeping some state for every connection (indexed list of previously sent headers)
- HPACK is based on static table of most commonly used headers, methods, statuses etc. (currently contains 61 items, "2" means "GET", "3" means "POST", "7" means "HTTPS", "28" means "Content-Length" etc.)
- This table can be dynamically extended to re-use the same mechanism for your custom site-specific headers
- Client can specify "priority hints", providing dependencies and importance of every request
- These hints can be updated at any point, so you can deprioritize some requests (e.g. after you downloaded the most important part, or if the user suddenly switches the tab)
- These are just hints, the server does not have to respect them
- HTTP/2 allows a server to populate data in a client cache through a mechanism called the **server push** (instead of waiting for a client to re-request a resource)
- Server push is somewhat experimental/questionable feature, not everyone likes it
- HTTP/2 does not require HTTPS, however, most browsers only implement HTTP/2 over HTTPS (which requires TLS 1.2+)
- The semantics of the messages is unchanged, hence it is useful to comprehend HTTP/2 messages in the HTTP/1.1 format
- HTTP/2 should be most useful in high latency networks or lots of requests to the same host, in that case you may expect 5-15% performance improvement by simply switching to HTTP/2
- The big issue with HTTP/2 is head-of-line blocking, related to lost TPC packages, as server needs to assemble these packets into the stream in order
- If a single TCP packet is lost, the server has to wait until the client re-sends the packet, with all the requests (streams) stalled
- Also, HTTP/2 can produce server spikes, and you might get the performance decrease when you enable it

### Outdated with HTTP/2

- Bundling CSS and JS files (into a single file with hash in a name and caching forever, at a cost of losing the granularity of caching)
- Domain sharding (use multiple subdomains to go above 6 parallel connections, at a cost of extra DNS lookups and connection overhead)
- Inlining (embedding CSS, images etc. within the HTML itself, many trade-offs, such as extra complexity)

### Still applicable

- Make fewer HTTP requests
- Send as little as possible
- Send it as infrequently as possible
- Minify
- Turn on compression
- Use browser cache!
- Use CDN


## QUIC and HTTP/3

- HTTP/3 is based on QUIC, which was another experiment at Google, announced in 2013, standardized in 2021
- The most important design goal is to reduce latency, especially for web applications
- The main mechanism is what we've already seen in HTTP/2: stream multiplexing
- Unfortunately, HTTP/2 came with its own problem: head-of-line blocking, that resulted from HTTP/2 being built on top of TCP
- To solve this problem, QUIC had to ditch the TCP (and basically, eliminate some abstraction layers to achieve better performance)
- So QUIC is build on top of UDP and takes over the responsibilities of TCP, TLS and partially application level (HTTP/2)
- QUIC is designed to obsolete TCP, so really think "TCP/2"
- QUIC could have really been built on top of IP, next to UDP and TCP, but that would require every device on Internet to know about the new protocol
- Same as TCP, QUIC is connection-oriented
- Unlike TCP, QUIC allows multiple streams on the same connection, and, unlike HTTP/2 over TCP, it is aware of multiple streams, and if an error occurs in one stream (e.g. lost packet), QUIC can continue servicing other streams independently
- QUIC implements its own loss recovery, retransmission, flow and congestion control
- There are extensions that allow disabling data retransmission, to make it more like UDP
- QUIC is also encrypted by design, so you don't need a separate layer, like TLS
- Additionally, packets are encrypted individually, which again, is possible, because QUIC is aware of packet boundaries within streams
- This means you don't have to wait for all packages in the stream to dencrypt the current package, which was another reason for head-of-line blocking in HTTP/2 over TCP implementation
- Another way to improve the latency is to reduce the overhead of the connection setup
- In pure TCP, you establish connection by doing 3-way handshake
- In TCP+TLS, this handshake gets extended with 3 more messages to negotiate the key; so you need to do 6-way exchange before you can finally start sending data
- With QUIC, you negotiate everything in just 3 messages, and the 3rd message already contains data
- Instead of [IP address + port] tuple to identify a connection, QUIC uses explicit connection id (each side of a connection has its connection id)
- This allows to keep the connection even if IP address changes (example: mobile device moves from a local WiFi hotspot to a mobile network)
- QUIC was developed with HTTP in mind, and HTTP/3 was its first application; but any other protocol can use QUIC
- HTTP/3 is basically about sending HTTP/2 frames over QUIC


## Cookies

- Cookies are the best current way to identify users and allow persistent sessions
- The cookie contains an arbitrary list of name=value information, and it is attached to the user using the Set-Cookie response header
- The only difference between session cookies and persistent cookies is when they expire
- A session cookie is deleted when the user exits the browser
- Persistent cookies are stored on disk and survive browser exits and computer restarts
- A server generating a cookie can control which sites get to see that cookie by adding a Domain attribute to the Set-Cookie response header
- Path attribute indicates the URL path prefix where each cookie is valid
- For more on cookies and web security, see [Computer Systems Security](security.md)


## Basic authentication

- In basic authentication, a web server can refuse a transaction, challenging the client for a valid username and password
- The server initiates the authentication challenge by returning a 401 status code instead of 200 and specifies the security realm being accessed with the WWW-Authenticate response header
- When the browser receives the challenge, it opens a dialog box requesting the username and password for this realm
- The username and password are sent back to the server in base-64 inside an Authorization request header (`Authorization: Basic YnJpYW...`)


## SSL

- HTTPS encrypts all message contents, including the HTTP headers and the request/response data
- Because SSL traffic is a binary protocol, completely different from HTTP, the traffic is carried on different ports
- Once the TCP connection is established, the client and server initialize the SSL layer, negotiating cryptography parameters and exchanging keys
- When the handshake completes, the SSL initialization is done, and the client can send request messages to the security layer


## Entities and transport

- HTTP also supports multipart bodies; however, they typically are sent in only one of two situations: in fill-in form submissions and in range responses carrying pieces of a document
- Form submission: provide your name, but also a photo
- Range requests: `Content-Type: multipart/byteranges` header and a multipart body with the different ranges
- Range Requests allows clients to request just part or a range of a document
- When content is dynamically created at a server, it may not be possible to know the length of the body before sending it; Chunked encoding provides a solution
- `Transfer-Encoding` header is used to tell the receiver that the message has been transfer-encoded with the chunked encoding
- Chunked encoding breaks messages into chunks of known size. Each chunk is sent one after another, eliminating the need for the size of the full message to be known before it is sent
- A client also may send chunked data to a server
