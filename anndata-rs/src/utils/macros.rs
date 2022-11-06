#[macro_export]
macro_rules! proc_numeric_data {
    ($dtype:expr, $reader:expr, $fun:ident, $ty:tt) => {
        match $dtype {
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U1) => {
                let mat: $ty<i8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U2) => {
                let mat: $ty<i16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U4) => {
                let mat: $ty<i32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U8) => {
                let mat: $ty<i64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U1) => {
                let mat: $ty<u8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U2) => {
                let mat: $ty<u16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
                let mat: $ty<u32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U8) => {
                let mat: $ty<u64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U4) => {
                let mat: $ty<f32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
                let mat: $ty<f64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Boolean => {
                let mat: $ty<bool> = $reader;
                $fun!(mat)
            },
            other => panic!("type {} is not supported", other),
        }
    }
}

#[macro_export]
macro_rules! proc_numeric_data_ref {
    ($dtype:expr, $reader:expr, $fun:ident, $ty:tt) => {
        match $dtype {
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U1) => {
                let mat: &$ty<i8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U2) => {
                let mat: &$ty<i16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U4) => {
                let mat: &$ty<i32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Integer(hdf5::types::IntSize::U8) => {
                let mat: &$ty<i64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U1) => {
                let mat: &$ty<u8> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U2) => {
                let mat: &$ty<u16> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U4) => {
                let mat: &$ty<u32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Unsigned(hdf5::types::IntSize::U8) => {
                let mat: &$ty<u64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U4) => {
                let mat: &$ty<f32> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
                let mat: &$ty<f64> = $reader;
                $fun!(mat)
            },
            hdf5::types::TypeDescriptor::Boolean => {
                let mat: &$ty<bool> = $reader;
                $fun!(mat)
            },
            other => panic!("type {} is not supported", other),
        }
    }
}

#[macro_export]
macro_rules! _box {
    ($x:expr) => { Ok(Box::new($x)) };
}