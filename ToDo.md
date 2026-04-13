### AI

- [ ] May be make a agent which can go through all the source files, and let us know if anything redundant, what can be removed, what can be optimized , what can be refactored etc.., because its very much required to maintain code quality as we are growing
- [ ] Ask AI to redocument every functions at once, and may be make some index, architecute.md that will be useful for it
- [ ] Update development guide so that its in sync with clang tidy

### Language / Structure Specific

- [ ] Have to fix the ABI , so that our ABI is independent of memory layout of caller, also update dev guide accordingly
- [ ] May be convert int type based enum to true C++ enums, and have conversion fns to convert from C style enum to Cpp enum, that way Cpp side api is more idiomatic
- [ ] Our ODAIDb (sqlite impl) is not thread safe, since db, statement, transaction, if at all we want to make it thread safe then one db object per thread, or use mutex locks
- [ ] Update defaults of structs, structs like Generator Config all needs to have good defaults
- [ ] Some functions like register model return false if model already registered , we need to properly express it with rich types

### Features / Bugs

- [ ] For model registration and update, may be can ask inference engine implementer to give a json schema for model files, and then we can validate it, may be validate model files really needs to be this
- [ ] Probably cache chat config in rag engine, instead of querying for every chat response generation
- [ ] Think on how to manage multiple chat sessions, may be LRU cache of chat sessions in backend engine, check mem constraints and handle accordingly
- [ ] Also see about serializing chat sessions to disk, so that way we can restore them later
- [ ] since we have removed contextWindowSize from config obj, and are using hardcoded ctx params size of 4096 for llm and mebedding models, think, Handle max tokens in llama context from sampler
- [ ] Add Structured Output Support
- [ ] Add a commit option in generating_streaming_chat_response, so that we can use it to try generate multiple answers without appending in chat history, useful for HYDE like thing 
- [ ] Currently we are not handling reasoning tokens separately, we are just using it as a normal token, think on how to handle it properly
- [ ] Update CreateSemanticSpace fn to auto infer dimensions from embedding model, also make sure to create necessary embedding model tables etc.., also update deleteSemanticSpace fn to delete the embedding model tables etc..
- [ ] For now everything is exposed via Public interface, later we will come up with a method so that people can just give Task Profile and then we will have a configuration for that task profile which we will use, making it simple
- [ ] Add RAG support
    - [ ] Implement simple Fixed Size Chunking Strategy
    - [ ] Store something in DB to identify which Chunking Strategy was used
    - [ ] Integrate embedding model to embed chunks and store in vector DB
    - [ ] Store something in DB to identify which embedding model was used to embed the documents
    - [ ] Implement vector storage and retrieval using sqlite vector extension
- [ ] Maybe can have a async variant of odai_generate_streaming_response
- [ ] Qualcomm Hexagon, QNN, Apple CoreML llama cpp integration pending
- [ ] when limited vram Think / explore on vision, audio encoder on GPU and LLM on cpu vs vice versa, both are useful for different workload


### Build System

- [ ] Update CMake so that it doesn't look for anything in system lib, instead everything should be built by ourselves, that way the build is portable
- [ ] Automate fetch and build of sqlite-vec also, like we have already did for sqlitecpp (& sqlite ) and llama
- [ ] Think on what would be the best way to structure your codebase, that is how to cross communicate JNI Libs and JNI cpp file
- [ ] Understand why android builds are resulting in large binary
- [ ] Have to make Pre-commit hook with clang efficient runnig scripts/lint.sh on all files is too much intensive


### Android / Mobile

- [ ] Be careful regarding Kotlin guarantees in CommonMain needn't transalte same on all targets, so think and verify
- [ ] Log stuff on KMP side
- [ ] Test if JNI Logging support is working
- [ ] May be makes sense to use some KMP Logging library
- [ ] After Library gets mature, think about thread safety in logging

### ToDo
- [ ] Reduce header dependency
- [ ] Figure out and ensure dotprod optimization is enabled for llamacpp (will have to check how to do)

### Tests
 - [ ] Test all GPU + optimization support
    - [ ] Test Windows / Linux with / without CUDA / HIP (discrete) / SYSCL (discrete) / Vulkan (iGPU)
    - [ ] For above also compare iGPU vs CPU, also check available RAM shown is correct or not
    - [ ] Test Windows / Linux ARM Laptops with / without VULKAN, also check if kleidiAI is enabled / used on devices , test Vulkan vs CPU
    - [ ] Test Windows / Linux lapotp are using optimal CPU for all variants
    - [ ] Test Android with / without vulkan, also check if dotprod and kleidiAI are enabled / used on devices
    - [ ] Test Apple devices with / without Metal, also check if dotprod and kleidiAI are enabled / used on devices