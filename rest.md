# REST Checklist
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [RESTful Web Services Cookbook by Subbu Allamaraju](https://www.oreilly.com/library/view/restful-web-services/9780596809140/)
- [REST in Practice by Jim Webber, Savas Parastatidis, and Ian Robinson](https://www.oreilly.com/library/view/rest-in-practice/9781449383312/)
- [REST API Design Rulebook by Mark Massé](https://www.oreilly.com/library/view/rest-api-design/9781449317904/)


## CRUD

### GET

- Is `GET` used to read a resource state?
- Is `GET` idempotent? (Can I repeat the same `GET` request?)
- Is `GET` safe? (Is `GET` free from side effects?)
- Does `GET` return `200 OK` when successful?
- Does `GET` return `404 Not Found` if resource does not exist?
- Are the resources that are often requested together combined into coarse-grained composites?
- Does resource granularity ensure that more cacheable, less frequently changing, or immutable data is separated from less cacheable, more frequently changing, or mutable data?
- Are similar resources organized into collections?

### POST

- Is `POST` used to create a new resource with a server-generated URI?
- Is a `Slug` header used to let clients suggest a name for the server to use as part of the URI of the new resource?
- Does `POST` return `201 Created` when successful?
- Is `Location` header used in the response to return a link to the new resource when successful?
- Are one-time URIs with generated tokens that are valid just for one usage used to prevent repeatable `POST`s?
- Does `POST` return `403 Forbidden` if the token was already used?

### PUT

- Is `PUT` used to update the resource state?
- Is `PUT` used to create a new resource with a client-generated URI?
- Is `PUT` idempotent? (Can I repeat the same `PUT` request?)
- Is `PUT` used to update the whole resource and not the part of it? (When you need to update the part of resource, you should use `PATCH` or `POST`. `PATCH` is not widely adopted yet, but it does exactly that.)
- Does `PUT` return `200 OK` or `204 No Content` when successful?
- Does `PUT` return `404 Not Found` if resource does not exist?
- Optional: does `PUT` create a new resource if it does not exist?
- Are entity tags (`ETag`) and conditional request headers (`If-Unmodified-Since`, `If-Match`) used to prevent concurrent update conflicts?
- Does `PUT` return `403 Forbidden` if the client does not include conditional request headers?
- Does `PUT` return `412 Precondition Failed` when conditional request headers do not match the current state?

### DELETE

- Is `DELET`E idempotent? (Can I repeat the same `DELETE` request?)
- Does `DELETE` return `204 No Content` when successful?
- Does `DELETE` return `404 Not Found` if resource does not exist?
- Does `DELETE` return `405 Method Not Allowed` if resource exists but cannot be deleted?
- Are entity tags (`ETag`) and conditional request headers (`If-Unmodified-Since`, `If-Match`) used to prevent deleting resources based on stale information?
- Does `DELETE` return `403 Forbidden` if the client does not include conditional request headers?
- Does `DELETE` return `412 Precondition Failed` when conditional request headers do not match the current state?

## Beyond CRUD

- Is `GET` used to support safe and idempotent computing/processing functions?
- Is `POST` used to perform any unsafe or nonidempotent operation that involves modifying more than one resource atomically, or whose mapping to `PUT` or `DELETE` is not obvious?
- Is `POST` used to start long-running requests asynchronously?
- Is `DELETE` used to start long-running resource deletion requests asynchronously?
- Are the custom HTTP methods used only when it’s absolutely impossible to avoid them?

## Representations

- Does server use `Content-Type` header to describe the type of the representation?
- Is `charset` parameter included into `Content-Type` header?
- Is `application/xml` media type (default UTF-8) used instead of `text/xml` (default us-ascii)?
- Optional: Are the content-type headers use application-specific media types (`application/vnd+xml`) to take advantage of hyper-media format?
- Does representation of the resource include a self link to the resource?
- Does representation of the resource include id for each of the application domain entities that make up a resource?
- If collection is paginated, does representation contain the link to the previous and the next page?
- Are the collections returned in the response homogenic?
- Is `XmlConvert` class used to properly encode data (dates and times, numbers etc.) in portable data formats?
- Is binary data retrieved using multipart media types or as a separate resource instead of encoding within textual formats using Base64 encoding?
- Is status code `4xx` used for errors due to client inputs?
- Is status code `5xx` used for errors due to server implementation or its current state?
- Optional: Does error response include an identifier or a link that can be used to refer to the error logged on the server side for later tracking or analysis?

## URIs

- Are singular nouns used for document names?
- Are plural nouns used for collection names?
- Are verbs or verb phrases used for controller names?
- Is media type used to indicate the representation format instead of file extension in URI?
- Is every URI designated to the unique resource instead of tunnelling requests to different resources through the same URI?
- Are the clients spared from constructing URIs by exposing only stable URIs and making all the other URIs discoverable from hyper-media links?
- Are URI cool? (Cool URIs don’t change.)