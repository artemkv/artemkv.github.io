# Computer Systems Security
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [MIT 6.858 Computer Systems Security 2014](https://www.youtube.com/playlist?list=PLUl4u3cNGP62K2DjQLRxDNRi0z2IRWnNh)
- [MIT 6.858 Computer Systems Security 2020](https://www.youtube.com/playlist?list=PLvyhBjPyU05rrodFofj_jBEyxFLhKoyeJ)

## Intro

- **Goals**: typically end-user data privacy and availability (despite the attack happening)
- **Policy**: the set of rules, e.g. only Alice can read file `F`, only Bob can update it
- **Threat model**: set of assumptions about an adversary, what they can do, e.g. password theft
- It's important to know where to draw the line (should you protect from the attacker breaking into your datacenter?), but in general, it's advisable to err on the side of caution
- **Mechanisms**: whatever software/hardware can help us to satisfy the policy
- You can never be sure you assembled the complete list of possible ways you can be attacked
- Defense is a so-called _negative goal_, so in general, there is no way to prove that you are safe
- Giants like Amazon and Apple made seemingly stupid mistakes in the past, so you are going to do them as well
- So in practice, this is the iterative process of continuous improvement
- Every system can be pushed to the point where it eventually breaks
- But don't despair, security is valuable even if not perfect
- At the end of the day, you are just making the cost of attack not worth it
- Unfortunately, in case of the Internet, the attacks are cheap
- You also need to think how you should recover if attack happens
- Security and usability are in constant conflict
- Amazon used to allow you buying stuff without logging in, which allowed using your CC on someone else's account, and then use that CC number to recover the password to that account
- Usually the main use case works very well, the things break on the periphery (e.g. password reset flow)
- Big source of security issues is insecure defaults (e.g. known default passwords, file uploaded in s3 being public by default)
- When having multiple components, verify that every component implements the policy. For example, rate limiting on login attempts; Apple forgot to implement it for the service of finding a missing phone, so attackers could use it to brute-force passwords and later go into other services (that had rate limiters)
- Don't assume your design/implementation is a secret, it is possible to reverse-engineer any system
- In fact, assume as little as possible (and document your assumptions, so you can review them)


## Security architecture

- [Google infrastructure security design overview](https://cloud.google.com/docs/security/infrastructure/design)
- **Isolation**: there should be boundaries across different "boxes" (services, VMs, host machines)
- Instead of one single boundary containing the whole system, it's better to have small boxes and nested boundaries around each of those (the least privilege principle); otherwise, once you get inside, you could do whatever you want
- The trend is to get smaller and smaller boundaries
- It gets increasingly popular to encrypt all the communication inside the cluster, while in the past people would be fine with terminating HTTPS at the boundary
- Can you really trust a datacenter you run in? Can you even trust the actual hardware that you have in the datacenter?
- At the same time, for pragmatic reasons, most systems do not have hard boundaries around each of the users. For example, there is no separate VM for each gmail user, which would be the most secure but also very costly (money and performance)
- By default, those boxes should not be able to get inside each other internals
- This is the safest, but eventually these boxes need to be able to work together, if we want to get anything useful out of our system
- **Sharing**: there should be a way for one box to share a resource with another box
- **Reference model**: "poke one hole in the boundary around a resource and put a guard in front of it"
- To make a decision, a guard needs to know the **Principal** (i.e. the identity of a caller) and **Policy** (what is allowed)
- For the damage control, the guard should also write an audit log
- The audit log should be stored in a separate service, otherwise it can be compromised when the service is compromised
- So the guard's main tasks are: **Authentication**, **Authorization**, and **Audit**
- To authenticate a request, it should come with some sort of **Credentials** (e.g. password)
- **Authentication** is, essentially, about mapping the credentials to a principal
- Somewhere there should be a table mapping principal names to their credential information (ideally, not in the guard itself, but in a separate service)
- **Authorization**: `permission = Policy(principal, resource)`
- So there should be somewhere another table, with principals as columns and resources as rows (again, ideally, in a separate service)
- If you store this information row-based (every principal's permissions for a given resource), you get **ACL** (typically stored together with the resource itself)
- If you store this information column-based (permissions for all resources for a given principal), you get **capabilities**
- Passing capabilities downstream removes the need to make a separate request to resolve permissions to a given resource every time you access a resource. The caller can simply present its capabilities to the callee
- In this sense, capabilities are similar to claims (but they are not the same)
- On the other hand, using capabilities make it difficult to answer the questions such as "who has access to this file?"
- So typically, capabilities are short-lived (e.g. 1 minute)
- **Audit**: log every decision you made in regard to authentication and authorization
- These logs are helpful to detect insider attacks

### Availability

- Essentially, availability boils down to defending from DoS attacks
- The most important plan when dealing with DoS attacks is having lots of resources
- Authenticate requests as soon as possible!
- Minimize resource usage before the request is authenticated


## User authentication

- 3 parts: registration, authentication check, recovery
- A lot of times, the requests are made on behalf of a user (by an agent, or another service), so you have **intermediate principals**
- This represents a challenge, since the attacker can make an intermediate principal act on her behalf
- Another challenge is establishing the identity. In many cases it's a case of **weak identity**: e.g. email, CC number at best
- Some services would require a strong registration process (banks), but in many cases, you want an Internet service to know as little as possible about you
- For this purpose, passwords have proven to be very user-friendly
- The big downside of a password is that they are extremely valuable
- At the same time there is an important human factor: humans are not capable of remembering strong passwords for each website
- The most popular password is "123456"
- Defense: rely on password as little as possible
- Use password only once per session
- Rate-limit use of passwords
- Augment passwords with a second factor (e.g. with SMS): helps with weak passwords and phishing attacks
- Time-Based One-Time Password Algorithm: [rfc6238](https://datatracker.ietf.org/doc/html/rfc6238)
- U2F (Universal 2nd Factor) is an open standard that strengthens and simplifies two-factor authentication using a device (USB or NFC)
- When user tries to log in, the server issues the challenge; all (almost all) the device is doing is signing the challenge with the private key of a user and passing it back to the server
- To guard against man-in-the-middle attack, the device receives the challenge together with the origin (the domain name) and TLS channel id (uniquely identifies TLS connection); so the device actually signs the challenge + origin + TLS channel id
- In this scheme the browser is trusted; if the browser is compromised, everything goes out of the window
- If you lose the device, you are pretty much fucked, that is why you are supposed to have a second device somewhere safe at home
- How much trust would you have in a manufacturer (and its supply chain)?
- Instead of storing passwords (in your DB), store hashes + salt (see notes of Cryptography for the details)


## Privilege separation (using micro-services)

- [Building Secure High-Performance Web Services with OKWS](https://css.csail.mit.edu/6.858/2022/readings/okws.pdf)
- This is about how to design systems to have security in the presence of bugs
- The idea is: if bugs are inevitable, the best remedy is to limit the effectiveness of attacks when they occur
- The main idea is to split the system into multiple smaller components that are isolated from each other
- Every component should run with the **least privilege**, meaning: be able to do strictly what it needs to do
- This reduces the surface of attack (as each component has less code, there are fewer bugs to exploit)
- This also limits the damage of an attack, if it happens
- This goes together with the **Segregation of duty**: no single principal should have too many privileges
- This raises the question on how to split the system into components, and how to share the data between the components
- Importantly, the security of the system should not come at the cost of its performance
- Case study: the Web server that powers OkCupid.com
- The site is split into multiple independent services: signin, matching, messaging, photos, profile editing, logging etc.
- The key assumption: the developers that work on these services are inexperienced in system programming/security
- Each service process is chrooted and runs as unprivileged user
- All the outside HTTP traffic goes to the OKD service ("OK dispatcher"), which then routes requests to appropriate services, over RPC
- In Unix, the port 80 is privileged (to bind to a privileged port, a process must be running with root permissions)
- This is quite bad, since the process that requires port 80 is also the one (and only one!) that is directly exposed to the external traffic
- OkCupid overcame this issue by using a separate process, OKLD ("launcher daemon"), that runs in the very beginning, with root permissions, sets all the other services up (including OKD), opens the connection on port 80, hands it to the OKD service and then stops
- This allows OKD service to run with very limited privileges
- Services cannot access to each other's DBs
- In fact, the services can't even go to their DBs directly
- Instead, they have to use the DB proxies that run on the DB machines
- Instead of issuing SQL statements directly, the services call specific functions of a DB proxy service over RPC
- All the SQL is constructed inside that proxy, written by very experienced, security-savvy programmers
- To prevent SQL injection attacks, all the SQL queries are known upfront and prepared when a proxy starts up; the runtime checks ensure that all parameters passed to those queries are appropriately escaped
- The DB proxy service also responsible for the access control
- OKLD also hands every service a token to access the DB proxy: some services (e.g. matching) have to look at the data from different users, so using the user token would not work
- In fact this is the reason this system could not have been designed to provide separation by user (instead of by service)

### Attack surface per service

- OKLD: if attacked, a disaster; but almost no surface (not connected to the Internet, has no API, and shuts down almost immediately)
- OKD: if attacked, could mess with HTTP traffic and call any other service, but could not get to the DB; the surface is narrow, since OKD only looks at the first line of an HTTP message to make the routing decision (no message parsing)
- Logging service: if attacked, could trash the logs, but not DB; the surface is a single RPC call (that allows to log a message)
- Matching service: if attacked, the data from multiple users may leak, but only the data that the service has access to, using a token; the surface is the set of RPC calls that the service exposes

### Isolation mechanisms

- You can run each service on separate physical machine, but that's expensive
- So better option is to run services on VMs, but there is a performance overhead
- There is much less overhead when you run multiple services on a same VM using containers
- Each container has its own file system, process id namespace, its own Ip address, and basically the only way to interact with a container is by establishing a network connection


## Linux containers

- Large attack surface with only 1 barrier to compromising the entire system is not a good way to build systems
- You want to break the attack service into small pieces and layer these attack services, so that full compromise of the system requires breaking through those multiple layers
- If you run anything under the `root`, you are just 1 barrier away from giving the full control to the attacker
- `chroot` changes the root directory of the current process and all of its child processes; however, if you are `root`, you can simply `chdir("../.")`
- One solution to this is sandboxing the running code, this way even if you are root, you are still limited
- **Type I virtualization**: Harware `|` Hyperviser `|` Guest OS
- **Type II virtualization**: Hardware `|` OS `|` Hyperviser `|` Guest OS
- **Containers**: Hardware `|` OS `|` Containers
- So, in a way, containers a kind of "chroot on steroids"
- Container is no more than just a process tree
- Container isolation relies on several Linux OS features: namespaces, cgroups, capabilities and Seccomp BPF
- **Namespaces** partition kernel resources such that one set of processes sees one set of resources, while another set of processes sees a different set of resources
- Namespaces: Process ID (pid), UTS (hostname, domain name), Network (net: IP addresses, routing table, socket listing etc.), Cgroup (see below), Inter-Process Communication (ipc: e.g. semaphores), User ID (user), Mount (mnt: mount points)
- You create a new process in a new namespace using `clone` (an upgraded version of the `fork` call)
- **Cgroups** allow fine-grained control over a process resource usage (memory, CPU, max open files etc.)
- **Capabilities** allow fine-grained control over the privileges that processes have on a Linux system (instead of just root/non-root)
- Essentially, you allow your process to do something that would require `root`, without actually running it as `root`
- The classical example is to give `CAP_NET_RAW` capability to `ping` util (to allow it creating raw network packets)
- Capabilities can be inherited
- Each capability normally groups several privileges, so it is not perfectly fine-grained, but still much better than just `root`
- **Seccomp (SECure COMPuting)** allows a process to make a one-way transition into a "secure" state where it cannot make any system calls except `exit()`, `sigreturn()`, `read()` and `write()` to already-open file descriptors
- This wasn't actually very useful
- **Seccomp BPF (SECure COMPuting with filters)**, a more recent extension to Seccomp provides a means for a process to specify a filter for system calls (and even system call arguments), minimizing the exposed kernel surface
- All the containers share the same kernel
- When you create a container, you need to specify an image
- An image contains base OS, system libraries, app dependencies and other
- So the root of a container FS actually looks like a root of OS FS
- Every container gets its own MAC address
- Network bridge is a device that has its own MAC address and allows 2 devices in 2 different collision domains talking to each other (operates on layer 2)
- To limit this communication, you can use `Iptables`: a software firewall for Linux distributions


## Software isolation (sandboxing)

- Can you do isolation without hardware/OS support?
- Even if you do have hardware/OS level, having another layer of isolation inside your software could create an extra layer of defense
- The main idea is sandboxing: i.e. containing untrusted code
- Many previous techniques are focused on keeping an attacker out; sandboxing, on the other hand, is about "keeping an attacker in the box"
- Some sandboxes are language-specific, e.g. executing the JS code inside a browser
- There exist language-independent sandbox solutions that work with native languages running directly on the CPU (important for high performance)
- For example, Google used to ship Chrome with the Native Client, a sandbox for running compiled C and C++ code in the browser; deprecated in 2020 in favor of WebAssembly

### Native Client

- [Native Client: A Sandbox for Portable, Untrusted x86 Native Code](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/34913.pdf)
- Deprecated in 2020 in favor of WebAssembly
- The flow: you write a program in C++, then compile it into the x86 code
- You then put it on your website so that it is downloaded by the browser
- Before you run the code, it passes through the validator; the job of the validator is to decide whether the code is safe to run
- If considered safe, x86 code then runs as a separate process (which is also limited, creating multiple layers of defense)
- The validator is, of course, a crucial piece of the whole implementation
- By design, the validator needs to be small, to reduce the attack surface (fewer lines of code → fewer bugs), so its code is less than 600 C statements
- In order to keep it this small, the functionality of a validator is limited purely to validation, and it relies on external compilation tools to do its job
- Turns out, to generate NaCl-compliant binary, you have to use modified GNU tool chain (not a standard compiler/assembler)
- However, the changes were minimal (less than 1000 lines of code), which claims to demonstrate "the simplicity of porting a compiler to Native Client"
- This approach allows to instrument some potentially unsafe instructions (`ld`\`st`) with extra checks at the compile time, so that validator only need to verify that those checks are in place
- Some instructions are always safe (`ADD`), so they are always allowed
- Some instructions are never safe: all syscalls (`INT 0x80`), `ret`; they are always disallowed
- Some instructions are rarely used and difficult to reason about, they are also disallowed (the modified compiler has a chance to replace them with something safe)
- Some instructions are "sometimes safe", so they need to be checked for safety
- Challenges: variable-length instructions, guarding `ld`/`st`/`jmp` instructions (no jumping in the middle of an instruction), entering and exiting the module (the sandboxed code needs to communicate with the rest of the code)
- To analyze the binary code, the validator needs to decode it first (so that it can verify all the `jmp` instruction targets etc.), this is about 1/3 of the whole validator code
- For any interaction with the browser (basically, any side effects), the module code has to transfer control to the runtime system code
- The runtime system is developed by NaCl team, shipped together with Chrome, and is mapped into the process memory
- The validator verifies that the code cannot jump into the runtime system directly
- Instead, the runtime system comes with the **trampoline**: a small block of trusted code that the program is allowed to jump to, and which then is allowed to call into the runtime
- This is the only way to transfer control to the runtime
- And to avoid tampering with the trampoline code, it is made read-only
- Conclusion: you can run 3rd party native code in the browser, sandboxed, with almost no performance penalty and minimal changes to the compilation toolchain; without relying on any hardware or OS support for isolation
- So why did Google deprecate it??? No one knows


## Enclaves

- [Komodo: Using verification to disentangle
secure-enclave hardware from software](https://css.csail.mit.edu/6.858/2020/readings/komodo.pdf)
- MIT actually removed this topic from 6.858 in 2022, but the ideas carry over to the mobile security topics
- The process isolation is based on virtual memory: every process has its own page table (the kernel manages that)
- Since the processes cannot access each other's memory, the way they interact with the kernel and other processes is through the system calls
- But what if your OS kernel is compromised (e.g. because of malware) or has critical bugs?
- Enclaves allow spawning processes outside the kernel control (and completely isolated and independent of the kernel and the rest of the processes)
- While kernel has no access to enclaves, the enclaves can call into kernel (e.g. to read a file)
- Model: Hardware `|` Monitor `|` Enclaves, [OS `|` processes]
- Monitor acts a bit like a hypervisor, but unlike hypervisor, the monitor is really thin
- Monitor is trusted
- All memory is divided into a secure region and an insecure region
- Monitor can access all memory
- The kernel can only access the insecure region
- Enclaves can access specific regions in the secure and insecure regions
- There are several proposals on how to implement this
- All proposals require specialized hardware to support this


## Mobile device security (based on iOS)

- [iOS Security](https://css.csail.mit.edu/6.858/2020/readings/ios-security-may19.pdf)
- Threat model: attacker steals the phone and wants to read the data
- Assumptions: the phone is locked and password protected (otherwise you are fucked)
- Potential attacks: exhaustive password search, remove the flash card and read the data directly, access the device over the network without unlocking, install your own OS on the device, etc.
- Main CPU runs OS, all the sensitive operations (verifying password, limiting the number of attempts to enter a password) run inside the enclave, on the enclave CPU
- All the communication from the sensors (fingerprint, faceid) to the enclave CPU is protected (encrypted and signed)
- Inside the enclave there is also UID: AES-256 bit key fused into the enclave chip during manufacturing
- No software or firmware can read UID directly (not even enclave CPU), encryption or decryption operations performed by dedicated AES engines implemented in silicon
- The enclave portion of the memory is encrypted with an ephemeral memory protection key that is entangled with UID and created at a device boot
- Entanglement is done by encrypting the user password with UID multiple times
- Secure boot begins with enclave CPU running code from the boot ROM (also found inside the enclave and is trusted)
- Boot ROM is small, since it's fixed, and Apple wants to be able to update as much as possible
- To verify that the rest of the code came from apple, boot ROM comes with Apple's public key burned in
- So every next piece of boot code checks that the next piece after that has come from Apple
- Eventually enclave passes control to the main CPU that runs iOS and apps
- So it's pretty difficult to install something on iPhone that is not properly signed
- However, this would not be enough to prevent an attacker downgrading the iOS version to the older one
- To protect against version downgrade, every iPhone has ECID: an ID of a device (stored in ROM), and Apple keeps track of all the updates you ever installed (the iPhone sends this data to Apple)
- So a lot of this relies on hardware, and presumably Apple designed their chips in a way that it is extremely difficult/expensive to cut them open and look inside
- Attack surface: Apple's private key, get into the chip and read UID, bugs in iOS/boot code


## Mobile app security (based on Android)

- [The Android Platform Security Model](https://css.csail.mit.edu/6.858/2020/readings/android-platform.pdf)
- In the world of desktop apps, users are principals, apps run with user privileges
- This means that the isolation is between different users of a desktop, but there is no isolation between apps
- This makes the data sharing easy: multiple apps can use the same set of files; you can open the same file with any app
- On the flip side, any app that user runs can do whatever that user can do (and without ever telling that user)
- In the world of web, isolation is between apps, and apps are fully sandboxed
- No app can access data from any other app, unless they are specifically designed to work together
- This is good for security, but is very limiting for data sharing: each pair of apps need to be connected explicitly, and this requires development effort
- In Android, apps are principals
- Android is build on a **multi-party consent model**: an action should only happen if all involved parties consent to it
- There are 3 parties: user, platform, and developer
- Android app is basically a process on Linux, paired with a **manifest**; signed by the private key of a developer
- An application stores its files in `/data/data/<app_name>`
- So 2 different apps would be 2 different processes, having access to 2 different folders
- OS uses mechanisms similar to containers to keep those processes isolated
- The data sharing is based on messages called **intents**
- The application sends the intent to the kernel, and the kernel passes that intent to the reference monitor
- The reference monitor is responsible for checking the intent and forwarding it to the correct recipient
- The main parts of an intent are: component name, action and data
- **Explicit intents** specify which component of which application will satisfy the intent, by specifying a full component name
- **Implicit intents** do not name a specific component, but instead declare a general action to perform, which allows a component from another app to handle it
- **Action** is the general action to be performed; one of the platform-defined values (e.g. `ACTION_DIAL`, `ACTION_MAIN` etc.)
- **Data** is the data to operate on, such as a person record in the contacts database, expressed as a Uri
- The receiver can specify which permissions the caller should have in order to be able to send intent to this receiver
- The caller specifies which permissions it needs in order to do its job
- The user may or may not agree on giving those permissions to the running app
- Permissions can be Normal, Runtime or Signature
- **Normal permissions** present very little risk to the user's privacy (`INTERNET`, `VIBRATE`, `MODIFY_AUDIO_SETTINGS`, etc.); these are mostly for information
- **Runtime permissions** (aka dangerous permissions) give your app access to restricted data or let your app perform restricted actions (`READ_CONTACTS`, `CAMERA`, `READ_SMS`, `RECORD_AUDIO` etc.); these trigger the user prompt
- **Signature permissions** is the one that can only be granted when the app is signed by the same certificate as the app or the OS that defines the permission (e.g. allow Facebook chat app to request the friend list from the Facebook main app)
- Apps define and request permissions in the manifest
- The receiver can also check the caller, and apply some extra custom logic to decide whether it should accept the intent
- Sometimes intent data contains sensitive information (`SMS_RECEIVED` has the actual SMS text); in this case you want to protect the content of intent and not the receiver
- In such a case, you can use broadcast intent, which allows specifying permission that receivers should have to be able to receive the intent


## Web Security Model

- [Mozilla Web Docs: HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [Mozilla Web Docs: Web Security](https://developer.mozilla.org/en-US/docs/Web/Security)
- [Mozilla Web Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)
- Today browsers are incredibly complicated and have an enormous attack surface
- Web pages contain content from many entities mixed together (ads, analytics, 3rd party libs like jQuery, HTML and JS, frames etc.)
- This raises the question: how can all these pieces interact? Can analytics access HTML? Can they access your JS code state?
- The common security model that governs these interactions is the **Same-origin policy** (essentially, trust by URI)
- In the most secure world, a browser would treat every URI as a separate protection domain and require explicit consent for content retrieved from one URI to interact with another URI
- This would be very cumbersome for both developers and users, this is why browsers group URIs together into larger protection domains called **origins**
- Origin: scheme (HTTP/S) + hostname ("foo.com") + port (80)
- So `http://example.com` and `https://example.com` are not the same origin; `https://example.com/aaa` and `https://example.com/bbb` are
- This division is not perfect, but there is a lot of history involved, and this is what web technologies have converged to in the end
- Each resource is assigned an origin, based on its URL
- Window/frame origin is determined based on its URL
- DOM nodes get the origin of the surrounding window/frame
- Now that we know how to determine the origin, we can consider who is allowed to do what
- Not every resource in an origin carries the same **authority** (that is to say they don't have the same permissions)
- Browsers are supposed (RFC6454) to determine the resource authority by examining its media type (e.g. `image/png` is treated as an image)
- Therefore, when hosting untrusted content (such as user-generated content), it could be a good idea for web applications to limit that content's authority by restricting its media type
- HTML document (`text/html`) carries the full authority of its origin
- JavaScript executes with the authority of its window/frame's origin (not its source)
- Passive content (images, CSS) carries no authority (meaning image has no access to the objects and resources available to its origin)
- An origin is permitted to execute script, render images, and apply style sheets from any origin
- An origin can also display content from another origin, such as an HTML document in an HTML frame
- While you can do all of that, inspecting the information retrieved from another origin is forbidden (browser will not allow you to access the results, unless you enable CORS)
- Note that this isolation is not perfect, as some information may leak, e.g. height and width of an image, CSS styles etc.
- An origin can send information to another origin, but agents restrict what is sent (e.g. don't send custom headers over HTTP)

### Frames

- Each frame gets the origin of its URL
- Frames can set their `document.domain` to the suffix of its URL (e.g. `x.y.z.com` -> `y.z.com`, but not `a.y.z.com`)
- 2 frames can access each other if either they both set their `document.domain` to the same value or neither of them has changes their `document.domain` and their values match
- If one of the conditions holds, 2 frames can interact without limitations
- Alternatively, 2 frames/windows can communicate by sending each other messages using `postMessage` interface (regardless of the origin)
- Frames can check whether they are top frames using `self === top`
- `X-Frame-Options` header allows prohibiting embedding content into a frame

### Cookies

- HTTP cookie is sent by a web server to a user's web browser, using `Set-Cookie` header
- The browser stores the cookie and sends it back to the same server with all later requests
- The cookies is just a name-value pair
- Cookies were once used for general client-side storage, nowadays, `localStorage` and `sessionStorage` are recommended instead
- So the cookies are still used for session management, personalization and behavior tracking
- Cookies have the scope, i.e. what URLs they should be sent to
- The scope is defined by the domain and path
- When domain is specified, then cookies are available on that domain and its subdomains; otherwise, only on the domain itself but not on its subdomains
- For the path, `%x2F` is used as a directory separator, and subdirectories match as well
- When path is not set, it is computed from the path of the URI that set the cookie (see the exact rules in the documentation)
- **Session cookies** are deleted when the current session ends
- **Permanent cookies** are deleted when expired (controlled by `Expires`/`Max-Age` attributes)
- If your site authenticates users, it should regenerate and resend session cookies, even ones that already exist, whenever a user authenticates (prevents session fixation attacks, when an attacker lures the victim to log in to a vulnerable site using attacker's SID)
- **Secure cookies** are only sent to the server over HTTPS (basically mandatory)
- **HttpOnly cookies** are inaccessible to JS, only sent to the server (recommended)
- **SameSite cookies** specify the behavior for cross-site requests
- With `SameSite=Strict`, the browser only sends the cookie with requests coming from the cookie's origin site (recommended, but consider "Lax" option as well)
- As a defense-in-depth measure, you can use cookie prefixes to assert specific facts about the cookie (e.g. `__Host-` for "domain-locked cookies")
- This can prevent session fixation attacks using cross-subdomain cookie
- There are main 2 vectors of attack: forcing your cookies on a user, or stealing the user's cookies
- If an attacker could set the cookies for a different origin, they would be able to make a victim authenticate as the attacker when going to a legit website
- Use HttpOnly cookies to prevent this
- When you fetch data from any URL, all the related cookies are sent along, so if an attacker could inject a script to victim.com, that script could, while executed in a user's browser, have authenticated access to all the domains the user has the cookies for (CSRF attack)
- CSRF attacks are prevented using a single use random token + SameSite=Strict

### Cross-Origin Resource Sharing (CORS)

- When doing AJAX GET requests to different origins, you cannot read the response by default
- When doing AJAX POST requests to different origins, only posting forms is allowed by default. To make sure posting a form is still safe, you need to protect from CSRF attacks (see below)
- Anything else is disallowed by default
- The server can allow all of that by using CORS
- To allow requests, server needs to indicate this using headers like `Access-Control-Allow-Origin`, `Access-Control-Allow-Credentials`, `Access-Control-Allow-Methods`, `Access-Control-Allow-Headers` etc.
- Browser will use these headers to decide what is allowed and what is not
- Browsers will also send `Origin` header along with request that servers could inspect
- To know whether the request is allowed or not before issuing an actual request, browser uses pre-flight requests
- Pre-flight request is a request to the same URL using `OPTIONS` method, with the goal of retrieving CORS headers to inspect
- "Simple requests" don't trigger pre-flight request. Simple requests are in essence: `GET`, `HEAD` and `POST` with form data
- Doing `POST` with "forbidden" headers, or using any other verbs (like `DELETE`, `PUT`, etc. regardless of the headers) will trigger pre-flight request
- This prevents issuing requests that might have side effects, like `DELETE` of the document before verifying that this request is allowed by CORS
- This is only safe when users access websites from the well-behaving browsers that respect CORS headers

### Content Security Policy (CSP)

- **Content Security Policy (CSP)** is an added layer of security that helps to detect and mitigate certain types of attacks, including Cross-Site Scripting (XSS) and data injection attacks
- A primary goal of CSP is to mitigate and report XSS attacks by disabling the use of unsafe inline JavaScript
- To enable CSP, you need to configure your web server to return the `Content-Security-Policy` HTTP header with the policy
- Alternatively, the `<meta>` element can be used to configure a policy
- The policy contains the set of directives and values that control what resources the user agent is allowed to load for that page
- Your policy should include a `default-src` policy directive, which is a fallback for other resource types when they don't have policies of their own
- `script-src` directive specifies valid sources for JavaScript
- `style-src` directive specifies valid sources for stylesheets
- `img-src` directive specifies valid sources of images and favicons
- Source can be a <host-source> (e.g. "http://*.example.com"), <scheme-source> (e.g. "http:" or "https:") or special kind of sources (e.g. `'self'`, `'unsafe-inline'`)
- `'self'` is a special value to refer to the current origin
- To specify that all content has to come from the site's own origin:

```
Content-Security-Policy: default-src 'self'
```

- By default, CSP disallows inline styles and inline scripts, which is one of the biggest advantages of using it
- You can bypass that by using `unsafe-eval`: a special value to refer to code that is produced by eval (works both for styles and scripts)
- Better way to do it (if you absolutely cannot move that code into a separate JS file) is by using `unsafe-hashes`, paired with the SHA-256 hash of that code (added in CSP Level 3)
- You can catch policy violations using `report-uri` directive
- `report-uri` directive allows to configure an endpoint where the information about violations is sent
- CSP report contains the resource URI, referrer, blocked URI, the violated directive etc.
- "https://report-uri.com/" is a service that you could use for receiving CSP reports and seeing nice charts

### Attacks

#### SQL Injection

- Mechanics: insert evil SQL statements into an entry field for execution
- Example:

```
txtUserId = getRequestString("UserId");
txtSQL = "SELECT * FROM Users WHERE UserId = " + txtUserId;
```

- Defense: use prepared statements with parameters

#### Cross-site Scripting (XSS)

- Mechanics: inject client-side scripts into web pages viewed by other users
- Example: submit this in a blog comment to make everyone who reads the blog see the alert:

```
<script>alert('you are hacked')</script>
```

- Defense: encode all your input

#### Cross-Site Request Forgery (CSRF)

- Mechanics: trick user into performing an action he or she didn't intend to do (especially in combination with XSS)

```
var xhr = new XMLHttpRequest();
xhr.open('DELETE', 'http://example.com/asset/123', true);
xhr.withCredentials = true;
xhr.send(null);
```

- Defense: introduce some randomness into URL, use anti-CSRF token, verify that request is valid before executing it


## Symbolic execution

- [EXE: Automatically Generating Inputs of Death](https://css.csail.mit.edu/6.858/2022/readings/exe.pdf)
- Symbolic execution is a way of executing a program abstractly in order to find bugs in the code
- Instead of running code on manually or randomly constructed input (like in case of manual tests and fuzzy tests), the program is run on symbolic input
- This is kind of like doing algebra with variables instead of numbers
- This way, when code runs, instead of running on one specific value of an input, it runs on all possible values of that input (and hence, explores all feasible program paths)
- The important fact here is that symbolic execution only explores feasible program paths, not all paths (which would lead to path explosion)
- To achieve that, the execution tracks the constraints on each symbolic variable (initially, no constraints), updating them as the code runs and reasons about all possible values of that variable
- Basically, any `if` statement splits the domain of a symbolic variable in 2, creating 2 different code paths
- These constraints allow detecting unreachable branches
- Any value that causes a bug is reported, and a test case is generated
- What is bug and how to recognize it?
- Easy cases: divide by 0, dereferencing null pointer, out-of-bound array access
- For application specific bugs, the developer needs to provide asserts that check invariants (so manual instrumentation is required)
- _My note: implementation details are not that important and hence omitted_


## Network security

- [A Look Back at “Security Problems in the TCP/IP Protocol Suite”](https://css.csail.mit.edu/6.858/2022/readings/lookback-tcpip.pdf)
- Threat model: attacker can intercept and modify packets, inject packets, participate in any protocol

### TCP Sequence Number Prediction

- TCP 3-way handshake: `SYN` → `SYN/ACK` → `ACK`
- `SYN` and `ACK` send sequence numbers along with the messages
- If you can predict server's sequence number, you could make a client `ACK` packet with a source IP of a trusted host and a correct sequence number, and impersonate that host
- The sequence numbers are random, so how can you predict that?
- Turns out, they are not completely random. As connections get re-used, you need to make sure previous connection packages arriving late don't get accidentally processed by the new connection
- To achieve that, in some systems, sequence numbers start random, but then just get incremented by some constant with every second
- So if you establish a legitimate connection once, you learn the current value of a sequence number, and then you can predict future sequence numbers based on time passed
- To avoid that the trusted host also sends an `ACK`, you could either flood it or just wait until it's offline
- The easy fix is to randomize the increment, but there are another solutions


## Messaging security

- [SoK: Secure Messaging](https://css.csail.mit.edu/6.858/2022/readings/secure-messaging.pdf)
- [SoK: Secure Messaging, extended version](https://css.csail.mit.edu/6.858/2020/readings/secure-messaging-ext.pdf)
- Setup: Alice wants to send message to Bob
- In many cases, messages, before getting delivered, go through multiple hops
- Most popular messaging tools used on the Internet do not offer end-to-end security
- End-to-end security relies on encryption and signing (make sure to sign message, not the ciphertext), this, however, does not protect from replay attacks (just capture the message, and send it again tomorrow)
- Problems: trust establishment (verifying that user is actually communicating with the party they intend), conversation security (confidentiality, integrity etc.), and transport privacy (hiding message metadata such as the sender, receiver etc.)
- The security schemes need to be easy-to-use and easy-to-adopt

### Trust establishment

- Most trust establishment schemes require key management: user agents must generate, exchange, and verify other participants' keys
- The baseline for evaluating any method of key exchange is **opportunistic encryption**: i.e. Alice just accepts Bob's public key
- This only protects against passive attackers (listen to the network, but don't try to inject anything), and it's super user-friendly since users don't have to do anything
- **Trust-on-first-use (TOFU)**: Alice accepts Bob's public key the first time she interacts with him (in-band), stores it, and uses it for all the future interactions
- With TOFU, there is no defined mechanism for key revocation; and there is still room to impersonate Bob (even though there is only 1 chance to do it)
- Still very user-friendly, since users still don't have to do anything
- **Out-of-band exchange**: for example, meet in person
- This is tedious, and you need to meet again if you want to use new key
- **Key server**: stores private keys and distributes public ones
- Here we are back to a perfect usability, and now you can do key revocation; so this system is widely used
- However, this introduces man-in-the-middle by design; also, stealing user password becomes as good as stealing the private key
- This idea can be extended with **key transparency**: basically, instead of storing the last private key in a table, do event sourcing (store every change in append-only fashion)
- This allows detecting key changes (Bob's device will see that the server has a new key, and Bob will be able to evaluate if it was him or not)
- Another extension is **keybase**: store user's extra handles next to their private key (Instagram, Twitter etc.)
- This allows Alice to confirm she is talking to the right Bob
- Keybase actually allowed writing, securely, to any twitter, reddit, facebook, etc. user, without knowing their email or phone
- Keybase was a really cool idea, but no one cared, so they shut down

### Conversation security

- This is actually hard, as there are many security and privacy features that you want to implement (Confidentiality, integrity, causality preserving, backward and forward secrecy and so on, see the paper)
- This is especially if we move into an area of group chats
- Some requirements are in tension: e.g. authenticity vs deniability
- Similarly, if messages are perfectly confident, you cannot run spam filters on them
- One of the interesting properties is deniability (see [Off-the-Record messaging](https://otr.cypherpunks.ca/index.php))
- Bob generates a random key `K`, encrypts using Alice's public key and sends to Alice
- Alice encrypts message `m` using Bob's public key, and sends it together with the authentication code, calculated using `K`
- All of this can be done by Bob using only public keys
- This allows Bob to prove to himself that it was Alice who sent him the message, but it does not allow Bob to prove that to the rest of the world (he could just have done it himself)
- Of course, you can prove this using out-of-band methods (e.g. ISP logs)
- The baseline approach is to use **trusted central servers** to relay messages over TLS (adopted by the most popular messaging systems today)
- This approach is very easy to implement, but it provides no end-to-end confidentiality
- **Static asymmetric cryptography** uses participants' static long-term asymmetric keypairs for signing and encrypting
- This approach has many fails to satisfy multiple requirements, such as forward or backward secrecy
- Other schemes are used each with their advantages and disadvantages

### Transport privacy

- Baseline is **Store-and-Forward** approach (basically, email)
- **Onion Routing** is a method for communicating through multiple proxy servers that complicates end-to-end message tracing
- **Dining Cryptographer networks (DC-nets)** are anonymity systems that achieve similar effect as onion routing
- **Broadcast Systems**: distributing messages to everyone. Sounds stupid, but actually provides recipient anonymity, participation anonymity, and unlinkability against all network attackers
- **Private Information Retrieval (PIR)** protocols allow a user to query a database on a server without enabling the server to determine what information was retrieved


## Private browsing

- [An Analysis of Private Browsing Modes in Modern Browsers](https://css.csail.mit.edu/6.858/2014/readings/private-browsing.pdf)
- Note: this is possibly very outdated, we are talking 2014 at best
- Goal #1: sites visited while browsing in private mode should leave no trace on the user's computer (protect from a local attacker)
- Threat model: local attacker (a family member) takes control of the machine after the user exits private browsing
- Things that browsers not allow leaking from a private browser session: cookies, local storage, browser cache, history
- Things that may still leak: configuration state (client certificates, bookmarks, passwords), maybe downloaded files, new plugins or extensions
- There are things that are outside of browser control that may leak: cached DNS records, OS paging files etc.
- Goal #2: hide user identity from websites they visit (protect from a web attacker)
- Threat model: an attacker has control over some visited sites
- We don't want the attacker to be able to link a user visiting in private mode to the same user visiting in public mode or in another private session
- We also don't want an attacker to be able to determine whether the browser is currently in private browsing mode
- What attacker can do is to try getting a browser fingerprint by running some JS on the client to extract all browser features, such as enumerating every installed plugin, installed fonts, time zone, screen size
- _My note: doing test using "coveryourtracks.eff.org", I get the following report: "Your browser fingerprint appears to be unique among the 183853 tested in the past 45 days"_
- There are tools to fingerprint your TCP protocol version just by sending you packets
- And even if you can't fingerprint a browser, you could fingerprint a user: users have unique keystroke timings and writing styles
- And then again, there are things outside a browser that can leak, such as IP address


## Anonymous Communication (Tor)

- [Tor: The Second-Generation Onion Router](https://css.csail.mit.edu/6.858/2014/readings/tor-design.pdf)
- Anonymity, as defined by Tor: inability of an observer on a network to link participants to actions
- Example: Alice buys socks from buysocks.com
- An observer should not even be able to say things like "Alice is more likely to have bought socks compared to everyone else"
- The actions are still visible, so it's known that someone bought socks
- The anonymity is based on the fact that there are billions of users; if there was only a bunch, Tor wouldn't help ("Anonymity loves company")
- Mechanics: Alice connects to a relay, the relay connects to all the sites users visit
- Besides Alice, every other user also connects to the same relay
- Everyone may run TLS, but the relay would still see who does what
- So let's have multiple relays, run by different people, each removing a layer of encryption
- The very first relay sees "Alice is doing something"
- The very last relay sees "Someone is buying socks"
- The relays in the middle see "Someone is doing something"
- To speed things up, the encryption is done using symmetric keys, not public keys
- In the beginning, Alice picks a circuit id, then negotiates the key `S1` with the first relay
- After that she sends an encrypted message to the first relay to extend the circuit
- The first relay then also picks some circuit id and negotiates the key `S2` with the second relay; returning that key to Alice
- Alice can now encrypt messages to buysocks.com first with `S2`, then with `S1`
- To achieve anonymity, you also need to obscure all the request sizes and timing of requests, otherwise, someone who eavesdrop on both ends could correlate messages; but with interactive network traffic and low latency it is difficult to achieve, so Tor cannot do much to prevent this kind of attacks
- The hard problem is discovering relays: anyone can run them, so if you ask relays, they could lie
- Having one centralized server would be a single point of failure
- The current solution (2014) is to have a number of trusted servers that agree with each other on the list of relays


## Secure Untrusted Data Repository (SUNDR)

- [Secure Untrusted Data Repository (SUNDR)](https://css.csail.mit.edu/6.858/2020/readings/sundr.pdf)
- SUNDR is a network file system designed to store data securely on untrusted servers
- It is a multi-user network file system that never presents applications with incorrect file system state, even when the server has been compromised (guarantees integrity)
- Server is assumed to be "Byzantine" and be able to collude with "bad users"
- The only assumption that is made is the attacker not being able to break the cryptography
- One of the benefits is ability to outsource storage management without fear of server operators tampering with data
- Typical use case is storing source code on 3rd party servers
- Unfortunately, SUNDR does not offer read protection or confidentiality
- All in all, this is just an academical research paper, and doesn't try to solve all the problems
- However, confidentiality can be achieved through encryption, and that is something that has been widely studied


## Data Tracking (TaintDroid)

- [TaintDroid: An Information-Flow Tracking System for Realtime Privacy Monitoring on Smartphones](https://css.csail.mit.edu/6.858/2014/readings/taintdroid.pdf)
- Goal: we want to prevent the sensitive data to escape the mobile device
- Example: malware reporting current user location, stealing the contact list, turn your phone into a bot for sending spam to all your contacts
- Main idea: track all the sensitive data (using taints) and stop it from being passed to the network calls
- **Sources** generate sensitive data (e.g. sensors, contact book)
- **Sinks** are the places that we don't want the tainted data to go (e.g. network)
- Use 32-bit vector to represent taint (meaning 32 possible sources)
- Then use a customized version of VM interpreter to track all the tainted data
