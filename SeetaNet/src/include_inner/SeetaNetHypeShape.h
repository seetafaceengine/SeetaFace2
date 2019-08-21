#ifndef _SEETANET_HYPESHAPE_H
#define _SEETANET_HYPESHAPE_H


#include <vector>

class HypeShape {
public:
    using self = HypeShape;
    using T = int32_t;

    explicit HypeShape( const std::vector<int> &shape )
        : m_shape( shape ) {
        // update weights
        if( m_shape.empty() ) return;
        m_weights.resize( m_shape.size() );
        auto size = m_shape.size();
        auto weight_it = m_weights.rbegin();
        auto shape_it = m_shape.rbegin();
        *weight_it++ = *shape_it++;
        for( size_t times = size - 1; times; --times ) {
            *weight_it = *( weight_it - 1 ) * *shape_it;
            ++weight_it;
            ++shape_it;
        }
    }

    T to_index( const std::initializer_list<T> &coordinate ) const {
        if( coordinate.size() == 0 ) return 0;
        auto size = coordinate.size();
        auto weight_it = m_weights.end() - size + 1;
        auto coordinate_it = coordinate.begin();
        T index = 0;
        for( size_t times = size - 1; times; --times ) {
            index += *weight_it * *coordinate_it;
            ++weight_it;
            ++coordinate_it;
        }
        index += *coordinate_it;
        return index;
    }

    T to_index( const std::vector<T> &coordinate ) const {
        if( coordinate.empty() ) return 0;
        auto size = coordinate.size();
        auto weight_it = m_weights.end() - size + 1;
        auto coordinate_it = coordinate.begin();
        T index = 0;
        for( size_t times = size - 1; times; --times ) {
            index += *weight_it * *coordinate_it;
            ++weight_it;
            ++coordinate_it;
        }
        index += *coordinate_it;
        return index;
    }

    T to_index( int arg0 ) {
        return arg0;
    }


#define LOOP_HEAD(n) constexpr size_t size = (n); auto weight_it = m_weights.end() - size + 1; T index = 0;
#define LOOP_ON(i) index += *weight_it * arg##i; ++weight_it;
#define LOOP_END(i) index += arg##i; return index;

    T to_index( int arg0, int arg1 ) {
        LOOP_HEAD( 2 )
        LOOP_ON( 0 )
        LOOP_END( 1 )
    }

    T to_index( int arg0, int arg1, int arg2 ) {
        LOOP_HEAD( 3 )
        LOOP_ON( 0 ) LOOP_ON( 1 )
        LOOP_END( 2 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3 ) {
        LOOP_HEAD( 4 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 )
        LOOP_END( 3 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4 ) {
        LOOP_HEAD( 5 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 )
        LOOP_END( 4 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5 ) {
        LOOP_HEAD( 6 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_END( 5 )
    }

    ///
    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6 ) {
        LOOP_HEAD( 7 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 )
        LOOP_END( 6 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7 ) {
        LOOP_HEAD( 8 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 )
        LOOP_END( 7 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7, int arg8 ) {
        LOOP_HEAD( 9 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 ) LOOP_ON( 7 )
        LOOP_END( 8 )
    }

    T to_index( int arg0, int arg1, int arg2, int arg3, int arg4,
                int arg5, int arg6, int arg7, int arg8, int arg9 ) {
        LOOP_HEAD( 10 )
        LOOP_ON( 0 ) LOOP_ON( 1 ) LOOP_ON( 2 ) LOOP_ON( 3 ) LOOP_ON( 4 )
        LOOP_ON( 5 ) LOOP_ON( 6 ) LOOP_ON( 7 ) LOOP_ON( 8 )
        LOOP_END( 9 )
    }

#undef LOOP_HEAD
#undef LOOP_ON
#undef LOOP_END

    std::vector<T> to_coordinate( T index ) const {
        if( m_shape.empty() )
            return std::vector<T>();
        std::vector<T> coordinate( m_shape.size() );
        to_coordinate( index, coordinate );
        return std::move( coordinate );
    }

    void to_coordinate( T index, std::vector<T> &coordinate ) const {
        if( m_shape.empty() ) {
            coordinate.clear();
            return;
        }
        coordinate.resize( m_shape.size() );
        auto size = m_shape.size();
        auto weight_it = m_weights.begin() + 1;
        auto coordinate_it = coordinate.begin();
        for( size_t times = size - 1; times; --times ) {
            *coordinate_it = index / *weight_it;
            index %= *weight_it;
            ++weight_it;
            ++coordinate_it;
        }
        *coordinate_it = index;
    }

    T count() const {
        return m_weights.empty() ? 1 : m_weights[0];
    }

    T weight( size_t i ) const {
        return m_weights[i];
    };

    const std::vector<T> &weight() const {
        return m_weights;
    };

    T shape( size_t i ) const {
        return m_shape[i];
    };

    const std::vector<T> &shape() const {
        return m_shape;
    };

    explicit operator std::vector<int>() const {
        return m_shape;
    }

private:
    std::vector<int32_t> m_shape;
    std::vector<T> m_weights;

public:
    HypeShape( const self &other ) = default;
    HypeShape &operator=( const self &other ) = default;

    HypeShape( self &&other ) {
        *this = std::move( other );
    }
    HypeShape &operator=( self &&other ) {
#define MOVE_MEMBER(member) this->member = std::move(other.member)
        MOVE_MEMBER( m_shape );
        MOVE_MEMBER( m_weights );
#undef MOVE_MEMBER
        return *this;
    }
};

#endif
