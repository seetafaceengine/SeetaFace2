//
// Created by kier on 2018/11/11.
//

#ifndef ORZ_TOOLS_CTXMGR_LITE_H
#define ORZ_TOOLS_CTXMGR_LITE_H

#include <thread>
#include <sstream>

#if defined(_MSC_VER) && _MSC_VER < 1900 // lower then VS2015
    #define ORZ_LITE_THREAD_LOCAL __declspec(thread)
#else
    #define ORZ_LITE_THREAD_LOCAL thread_local
#endif

namespace seeta
{

    namespace orz
    {
        class NoLiteContextException : public std::logic_error {
        public:
            NoLiteContextException()
                : NoLiteContextException( std::this_thread::get_id() ) {
            }

            explicit NoLiteContextException( const std::thread::id &id )
                : logic_error( build_message( id ) ), m_thread_id( id ) {
            }

        private:
            std::string build_message( const std::thread::id &id ) {
                std::ostringstream oss;
                oss << "Empty context in thread: " << id;
                return oss.str();
            }

            std::thread::id m_thread_id;
        };

        template<typename T>
        class __thread_local_lite_context {
        public:
            using self = __thread_local_lite_context;

            using context = void *;

            static context swap( context ctx );

            static void set( context ctx );

            static const context get();

            static const context try_get();

        private:
            static ORZ_LITE_THREAD_LOCAL context m_ctx;
        };

        template<typename T>
        class __lite_context {
        public:
            using self = __lite_context;
            using context = void *;

            explicit __lite_context( context ctx );

            ~__lite_context();

            static void set( context ctx );

            static context get();

            static context try_get();

            __lite_context( const self & ) = delete;

            self &operator=( const self & ) = delete;

            context ctx();

            const context ctx() const;

        private:
            context m_pre_ctx = nullptr;
            context m_now_ctx = nullptr;
        };

        namespace ctx
        {
            namespace lite
            {
                template<typename T>
                class bind {
                public:
                    using self = bind;

                    explicit bind( T *ctx )
                        : m_ctx( ctx ) {
                    }

                    explicit bind( T &ctx_ref )
                        : bind( &ctx_ref ) {
                    }

                    ~bind() = default;

                    bind( const self & ) = delete;

                    self &operator=( const self & ) = delete;

                private:
                    __lite_context<T> m_ctx;
                };

                template<typename T>
                inline T *get()
                {
                    return reinterpret_cast<T *>( __lite_context<T>::try_get() );
                }

                template<typename T>
                inline T *ptr()
                {
                    return reinterpret_cast<T *>( __lite_context<T>::try_get() );
                }

                template<typename T>
                inline T &ref()
                {
                    return *reinterpret_cast<T *>( __lite_context<T>::get() );
                }

                template<typename T, typename... Args>
                inline void initialize( Args &&...args )
                {
                    auto ctx = new T( std::forward<Args>( args )... );
                    __lite_context<T>::set( ctx );
                }

                template<typename T>
                inline void finalize()
                {
                    delete ptr<T>();
                }

                template<typename T>
                class bind_new {
                public:
                    using self = bind_new;

                    template<typename... Args>
                    explicit bind_new( Args &&...args )
                        : m_ctx( new T( std::forward<Args>( args )... ) ) {
                        m_object = m_ctx.ctx();
                    }

                    ~bind_new() {
                        delete m_object;
                    }

                    bind_new( const self & ) = delete;

                    self &operator=( const self & ) = delete;

                private:
                    __lite_context<T> m_ctx;
                    T *m_object;
                };
            }
        }
    }

}
using namespace seeta;

#endif //ORZ_TOOLS_CTXMGR_LITE_H
