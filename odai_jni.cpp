#include <jni.h>
#include <string>
#include <vector>

#include "odai_public.h"

typedef struct
{
  JNIEnv* m_env;             // valid because call is synchronous
  jobject m_callback_ref;     // global or local ref
  jmethodID m_on_chunk_method; // cached method ID
} StreamingResponseCallbackContext;

typedef struct
{
  JNIEnv* m_env;           // valid because call is synchronous
  jobject m_callback_ref;   // global or local ref
  jmethodID m_on_log_method; // cached method ID
} LogCallbackContext;

// -----------------------------------------------------------------------------
// Helper: Convert jstring (Java) to std::string (C++)
// -----------------------------------------------------------------------------
std::string jstring_to_string(JNIEnv* env, jstring j_str)
{
  if (!jStr)
    return "";
  const char* chars = env->GetStringUTFChars(jStr, nullptr);
  std::string ret(chars);
  env->ReleaseStringUTFChars(jStr, chars);
  return ret;
}

void log_callback(OdaiLogLevel log_level, const char* msg, void* user_data)
{
  LogCallbackContext* ctx = (LogCallbackContext*)user_data;
  JNIEnv* env = ctx->env;

  jstring j_msg = env->NewStringUTF(msg);

  env->CallVoidMethod(ctx->callbackRef, ctx->onLogMethod, jMsg);

  env->DeleteLocalRef(jMsg);
}

bool streaming_callback(const char* chunk, void* user_data)
{
  StreamingResponseCallbackContext* ctx = (StreamingResponseCallbackContext*)user_data;
  JNIEnv* env = ctx->env;

  jstring j_chunk = env->NewStringUTF(chunk);

  jboolean keep_going = env->CallBooleanMethod(ctx->callbackRef, ctx->onChunkMethod, jChunk);

  env->DeleteLocalRef(jChunk);

  return keepGoing ? true : false;
}

extern "C"
{

  // It's not recommended to set logger for ddai library for android,
  // only useful for debugging purpose, because in production it will be too many callbacks for logs

  JNIEXPORT void jnicall Java_com_odai_demo_RagEngine_odaiSetLogger(JNIEnv* env, jobject thiz, jobject callbackObj)
  {

    LogCallbackContext ctx;

    ctx.env = env;
    ctx.callbackRef = callbackObj;

    jclass callbackClass = env->GetObjectClass(callbackObj);

    // Assuming: fun onLog(logLevel: Int, msg: String)
    jmethodID onLogMethod = env->GetMethodID(callbackClass, "onLog", "(ILjava/lang/String;)Z");

    if (onLogMethod == nullptr)
    {
      return;
    }

    ctx.onLogMethod = onLogMethod;

    odai_set_logger(log_callback, &ctx);
  }

  JNIEXPORT void jnicall Java_com_odai_demo_RagEngine_odaiSetLogLevel(OdaiLogLevel log_level)
  {
    odai_set_log_level(log_level);
  }

  JNIEXPORT jboolean JNICALL Java_com_odai_demo_RagEngine_odaiInitializeSDK(JNIEnv* env, jobject thiz, jstring dbPath,
                                                                            jint backendEngineType)
  {

    const char* cDbPath = env->GetStringUTFChars(dbPath, nullptr);

    bool result = odai_initialize_sdk(cDbPath, backendEngineType);

    return result ? JNI_TRUE : JNI_FALSE;
  }

  JNIEXPORT jboolean JNICALL Java_com_odai_demo_RagEngine_odaiInitializeRagEngine(JNIEnv* env, jobject thiz,
                                                                                  jstring embeddingModelPath,
                                                                                  jstring llmModelPath)
  {

    const char* cEmbedPath = env->GetStringUTFChars(embeddingModelPath, nullptr);
    const char* cLlmPath = env->GetStringUTFChars(llmModelPath, nullptr);

    struct i_RagConfig config;

    config.embeddingModelConfig.modelPath = cEmbedPath;

    config.llmModelConfig.modelPath = cLlmPath;

    bool result = odai_initialize_rag_engine(&config);

    env->ReleaseStringUTFChars(embeddingModelPath, cEmbedPath);
    env->ReleaseStringUTFChars(llmModelPath, cLlmPath);

    return result ? JNI_TRUE : JNI_FALSE;
  }

  JNIEXPORT jboolean JNICALL Java_com_odai_demo_RagEngine_odaiAddDocument(JNIEnv* env, jobject thiz, jstring content,
                                                                          jstring documentId, jstring scopeId)
  {

    return JNI_TRUE;
  }

  // -----------------------------------------------------------------------------
  // 4. odaiGenerateResponse
  // -----------------------------------------------------------------------------
  JNIEXPORT jstring JNICALL Java_com_odai_demo_RagEngine_odaiGenerateResponse(JNIEnv* env, jobject thiz, jstring query,
                                                                              jstring scopeId)
  {

    const char* cQuery = env->GetStringUTFChars(query, nullptr);
    const char* cScopeId = env->GetStringUTFChars(scopeId, nullptr);

    char buf[2048];
    int32_t ret_size = odai_generate_response(cQuery, cScopeId, buf, sizeof(buf));

    return env->NewStringUTF(buf);
  }

  // -----------------------------------------------------------------------------
  // 5. odaiGenerateStreamingResponse
  // -----------------------------------------------------------------------------
  JNIEXPORT jboolean JNICALL Java_com_odai_demo_RagEngine_odaiGenerateStreamingResponse(JNIEnv* env, jobject thiz,
                                                                                        jstring query, jstring scopeId,
                                                                                        jobject callbackObj)
  {

    StreamingResponseCallbackContext ctx;

    ctx.env = env;
    ctx.callbackRef = callbackObj;

    const char* cQuery = env->GetStringUTFChars(query, nullptr);
    const char* cScopeId = env->GetStringUTFChars(scopeId, nullptr);

    // 1. Get the class of the callback object
    jclass callbackClass = env->GetObjectClass(callbackObj);

    // 2. Get method ID.
    // WARNING: Ensure the method name and signature match your Kotlin interface exactly.
    // Assuming: fun onToken(chunk: String) -> void
    jmethodID onChunkMethod = env->GetMethodID(callbackClass, "onChunk", "(Ljava/lang/String;)Z");

    if (onChunkMethod == nullptr)
    {
      return JNI_FALSE; // Method not found
    }

    ctx.onChunkMethod = onChunkMethod;

    int32_t ret_size = odai_generate_streaming_response(cQuery, cScopeId, streaming_callback, &ctx);

    if (ret_size < 0)
      return JNI_FALSE;

    return JNI_TRUE;
  }

} // extern "C"