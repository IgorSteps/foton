//#pragma once
//
//class Image {
//public:
//	Image() 
//	{
//		// Calculate the image height, and ensure that it's at least 1.
//		m_ImageHeight = static_cast<int>(m_ImageWidth / m_AspectRatio);
//		m_ImageHeight = (m_ImageHeight < 1) ? 1 : m_ImageHeight;
//	}
//
//	int Height() const 
//	{
//		return m_ImageHeight;
//	}
//
//	int Width() const
//	{
//		return m_ImageWidth;
//	}
//
//private:
//	const float m_AspectRatio = 16.0f / 9.0f;
//	const int m_ImageWidth = 400;
//	int m_ImageHeight;
//};