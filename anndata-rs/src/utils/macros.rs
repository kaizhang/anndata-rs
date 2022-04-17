/*
macro_rules! proc_scalar_data {
    ($dtype:expr, $reader:expr, $fun:ident) => {
        match dtype {
            DataType::Scalar(Integer(IntSize::U1)) => Ok(Box::new(Scalar::<i8>::read(container)?)),
            DataType::Scalar(Integer(IntSize::U2)) => Ok(Box::new(Scalar::<i16>::read(container)?)),
            DataType::Scalar(Integer(IntSize::U4)) => Ok(Box::new(Scalar::<i32>::read(container)?)),
            DataType::Scalar(Integer(IntSize::U8)) => Ok(Box::new(Scalar::<i64>::read(container)?)),

            DataType::Scalar(Unsigned(IntSize::U1)) => Ok(Box::new(Scalar::<u8>::read(container)?)),
            DataType::Scalar(Unsigned(IntSize::U2)) => Ok(Box::new(Scalar::<u16>::read(container)?)),
            DataType::Scalar(Unsigned(IntSize::U4)) => Ok(Box::new(Scalar::<u32>::read(container)?)),
            DataType::Scalar(Unsigned(IntSize::U8)) => Ok(Box::new(Scalar::<u64>::read(container)?)),

            DataType::Scalar(Float(FloatSize::U4)) => Ok(Box::new(Scalar::<f32>::read(container)?)),
            DataType::Scalar(Float(FloatSize::U8)) => Ok(Box::new(Scalar::<f64>::read(container)?)),

            DataType::Scalar(VarLenUnicode) => Ok(Box::new(String::read(container)?)),
            DataType::Scalar(Boolean) => Ok(Box::new(Scalar::<bool>::read(container)?)),
        }
    };
}
*/

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

macro_rules! _box {
    ($x:expr) => { Ok(Box::new($x)) };
}

pub(crate) use proc_numeric_data;
pub(crate) use _box;