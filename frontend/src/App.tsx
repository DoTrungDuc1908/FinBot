import React, { useState, useEffect, useRef } from 'react';
import { 
  Menu, Plus, MessageSquare, Settings, HelpCircle, Send, 
  Paperclip, Mic, Sparkles, History, Compass, Lightbulb, Code, TrendingUp, Calendar
} from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { 
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend 
} from 'recharts';
import './index.css';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  latency_ms?: number;
  chart_metadata?: any; 
}

const API_BASE_URL = 'http://localhost:8080';

const INDICATOR_OPTIONS = [
  { id: 'sma', label: 'SMA' },
  { id: 'rsi', label: 'RSI' },
  { id: 'macd', label: 'MACD' },
  { id: 'bbands', label: 'Bollinger Bands' }
];

const StockChart: React.FC<{ metadata: any }> = ({ metadata }) => {
  const [data, setData] = useState<any[]>([]);
  const [period, setPeriod] = useState(metadata.default_period || '6m');
  const [loading, setLoading] = useState(false);
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>([metadata.default_indicator || 'sma']);

  const toggleIndicator = (ind: string) => {
    setSelectedIndicators(prev => {
      if (prev.includes(ind)) {
        if (prev.length === 1) return prev;
        return prev.filter(i => i !== ind);
      }
      return [...prev, ind];
    });
  };

  useEffect(() => {
    const fetchChartData = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`${API_BASE_URL}/stock/${metadata.ticker}/technical`, {
          params: {
            indicator: selectedIndicators.join(','),
            period: metadata.start_date ? undefined : period,
            start: metadata.start_date,
            end: metadata.end_date,
            interval: metadata.interval || '1D',
            full_data: true 
          }
        });
        if (res.data && res.data.data && res.data.data.history_data) {
          setData(res.data.data.history_data);
        }
      } catch (error) {
        console.error("Lỗi khi tải dữ liệu biểu đồ", error);
      } finally {
        setLoading(false);
      }
    };
    fetchChartData();
  }, [metadata.ticker, selectedIndicators, period, metadata.start_date, metadata.end_date, metadata.interval]);

  const isCustomDate = metadata.start_date && metadata.end_date;

  return (
    <div style={{ marginTop: '20px', backgroundColor: 'var(--bg-sidebar)', padding: '20px', borderRadius: '16px', width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '20px' }}>
        
        {/* Header & Chọn thời gian */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: '16px', color: 'var(--accent)' }}>
            Biểu đồ Phân tích Kỹ thuật - {metadata.ticker.toUpperCase()} {metadata.interval ? `(${metadata.interval})` : ''}
          </h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            {isCustomDate ? (
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'var(--bg-hover)', padding: '6px 14px', borderRadius: '8px' }}>
                {metadata.start_date} → {metadata.end_date}
              </span>
            ) : (
              ['1m', '3m', '6m', '1y'].map(p => (
                <button key={p} onClick={() => setPeriod(p)}
                  style={{
                    background: period === p ? 'var(--accent)' : 'var(--bg-hover)',
                    color: period === p ? 'var(--bg-main)' : 'var(--text-primary)',
                    border: 'none', padding: '6px 14px', borderRadius: '8px',
                    cursor: 'pointer', fontSize: '12px', fontWeight: 600, transition: 'all 0.2s'
                  }}>
                  {p.toUpperCase()}
                </button>
              ))
            )}
          </div>
        </div>

        {/* Nút Toggle Chọn Chỉ Báo */}
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {INDICATOR_OPTIONS.map(opt => (
            <button key={opt.id} onClick={() => toggleIndicator(opt.id)}
              style={{
                background: selectedIndicators.includes(opt.id) ? 'var(--accent)' : 'transparent',
                color: selectedIndicators.includes(opt.id) ? 'var(--bg-main)' : 'var(--accent)',
                border: '1px solid var(--accent)', padding: '4px 12px', borderRadius: '16px',
                cursor: 'pointer', fontSize: '12px', fontWeight: 600, transition: 'all 0.2s'
              }}>
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ height: '400px', width: '100%' }}>
        {loading ? (
          <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
            Đang tính toán chỉ báo...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="date" stroke="var(--text-secondary)" fontSize={12} tickMargin={10} minTickGap={30} />
              
              {/* Trục Y chính (Giá, SMA, BBands) */}
              <YAxis yAxisId="left" stroke="var(--text-secondary)" fontSize={12} domain={['auto', 'auto']} />
              
              {/* Trục Y phụ (RSI: 0 - 100) */}
              {selectedIndicators.includes('rsi') && (
                <YAxis yAxisId="right_rsi" orientation="right" stroke="#f28b82" fontSize={12} domain={[0, 100]} />
              )}
              
              {/* Trục Y phụ 2 (MACD: Âm / Dương) */}
              {selectedIndicators.includes('macd') && (
                <YAxis yAxisId="right_macd" orientation="right" stroke="#fbbc04" fontSize={12} domain={['auto', 'auto']} />
              )}

              <RechartsTooltip 
                contentStyle={{ backgroundColor: 'var(--bg-main)', border: '1px solid var(--border)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }} labelStyle={{ color: 'var(--text-secondary)', marginBottom: '4px' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              
              <Line yAxisId="left" type="monotone" dataKey="close" name="Giá đóng cửa" stroke="#e3e3e3" strokeWidth={2} dot={false} />
              
              {selectedIndicators.includes('sma') && (
                <Line yAxisId="left" type="monotone" dataKey="sma" name="SMA" stroke="var(--accent)" strokeWidth={2} dot={false} />
              )}
              
              {selectedIndicators.includes('bbands') && (
                <>
                  <Line yAxisId="left" type="monotone" dataKey="upper_band" name="BB Upper" stroke="#ccff90" strokeDasharray="3 3" strokeWidth={1} dot={false} />
                  <Line yAxisId="left" type="monotone" dataKey="middle_band" name="BB Mid" stroke="#ccff90" strokeWidth={1.5} dot={false} />
                  <Line yAxisId="left" type="monotone" dataKey="lower_band" name="BB Lower" stroke="#ccff90" strokeDasharray="3 3" strokeWidth={1} dot={false} />
                </>
              )}

              {selectedIndicators.includes('rsi') && (
                <Line yAxisId="right_rsi" type="monotone" dataKey="rsi" name="RSI" stroke="#f28b82" strokeWidth={2} dot={false} />
              )}

              {selectedIndicators.includes('macd') && (
                <>
                  <Bar yAxisId="right_macd" dataKey="histogram" name="MACD Hist" fill="#fbbc04" opacity={0.5} />
                  <Line yAxisId="right_macd" type="monotone" dataKey="macd" name="MACD" stroke="#fbbc04" strokeWidth={2} dot={false} />
                  <Line yAxisId="right_macd" type="monotone" dataKey="signal_line" name="Signal" stroke="#ff8a65" strokeWidth={2} strokeDasharray="3 3" dot={false} />
                </>
              )}

            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};


const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [marketOverview, setMarketOverview] = useState<any>(null);
  
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [interval, setInterval] = useState('1D');

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchMarketOverview();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  const fetchMarketOverview = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/market`);
      setMarketOverview(response.data.data);
    } catch (error) {
      console.error('Error fetching market:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        question: input,
        risk_profile: 'trung bình',
        session_id: 'browser-user-' + Date.now(),
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        interval: interval
      });

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.data.answer,
        timestamp: new Date(),
        latency_ms: response.data.latency_ms,
        chart_metadata: response.data.chart_metadata
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'Xin lỗi, đã có lỗi xảy ra khi kết nối với máy chủ. Vui lòng thử lại sau.',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const renderMarketData = () => {
    if (!marketOverview) return null;
    
    let marketItems = [];
    if (Array.isArray(marketOverview)) {
      marketItems = marketOverview.slice(0, 3).map((data: any, idx: number) => ({
        name: data.ticker || data.index || `Index ${idx}`,
        price: data.price || data.close || '---',
        change: data.change || data.percent_change || 0
      }));
    } else {
      marketItems = Object.entries(marketOverview).slice(0, 3).map(([key, data]: [string, any]) => ({
        name: key,
        price: data.price || '---',
        change: data.change || 0
      }));
    }

    return marketItems.map((item, index) => (
      <button key={index} className="chat-item">
        <TrendingUp size={16} />
        <span style={{flex: 1}}>{item.name}</span>
        <span style={{fontSize: '12px', color: item.change >= 0 ? '#10b981' : '#ef4444'}}>
          {item.change >= 0 ? '+' : ''}{item.change}%
        </span>
      </button>
    ));
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className={`sidebar ${!isSidebarOpen ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <button className="menu-btn" onClick={() => setIsSidebarOpen(false)}>
            <Menu size={24} />
          </button>
        </div>
        
        <button className="new-chat-btn" onClick={() => setMessages([])}>
          <Plus size={20} />
          Đoạn chat mới
        </button>

        <div className="recent-chats">
          <div className="recent-title">Thị trường</div>
          {renderMarketData()}
          
          <div className="recent-title" style={{ marginTop: '24px' }}>Gần đây</div>
          <button className="chat-item"><MessageSquare size={16} />Phân tích cổ phiếu VNM</button>
          <button className="chat-item"><MessageSquare size={16} />Giá HPG hôm nay thế nào?</button>
        </div>

        <div className="sidebar-footer">
          <button className="footer-btn"><HelpCircle size={20} />Trợ giúp</button>
          <button className="footer-btn"><History size={20} />Hoạt động</button>
          <button className="footer-btn"><Settings size={20} />Cài đặt</button>
        </div>
      </aside>

      {/* Main Area */}
      <main className="main-area">
        <header className="main-header">
          <div className="header-left">
            {!isSidebarOpen && (
              <button className="menu-btn" onClick={() => setIsSidebarOpen(true)}>
                <Menu size={24} />
              </button>
            )}
            <div className="model-selector">
              FinBot <span style={{fontSize: '14px', color: 'var(--text-secondary)'}}>Llama 3.1</span>
            </div>
          </div>
          <div className="header-right">
            <div className="avatar">D</div>
          </div>
        </header>

        <div className="chat-container" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="greeting">
                <div className="greeting-hello">Xin chào,</div>
                <div className="greeting-sub">Tôi có thể giúp gì cho bạn hôm nay?</div>
              </div>

              <div className="suggestion-cards">
                <button className="suggestion-card" onClick={() => setInput("Phân tích kỹ thuật cổ phiếu FPT")}>
                  <div>Tạo báo cáo<br/>phân tích kỹ thuật cho FPT</div>
                  <div className="card-icon"><Compass size={24} /></div>
                </button>
                <button className="suggestion-card" onClick={() => setInput("Xu hướng thị trường chứng khoán Việt Nam")}>
                  <div>Tóm tắt tin tức<br/>thị trường hôm nay</div>
                  <div className="card-icon"><Lightbulb size={24} /></div>
                </button>
                <button className="suggestion-card" onClick={() => setInput("Định giá cổ phiếu HPG")}>
                  <div>Lập kế hoạch<br/>đầu tư dài hạn</div>
                  <div className="card-icon"><TrendingUp size={24} /></div>
                </button>
                <button className="suggestion-card" onClick={() => setInput("Phân tích RSI cổ phiếu VIC")}>
                  <div>Phân tích kỹ thuật<br/>chuyên sâu</div>
                  <div className="card-icon"><Code size={24} /></div>
                </button>
              </div>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg) => (
                <div key={msg.id} className={`message ${msg.type}`}>
                  {msg.type === 'bot' && <Sparkles className="bot-icon" />}
                  <div style={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                    {/* Render nội dung tin nhắn dưới định dạng Markdown */}
                    <div className="message-content" style={{ overflowWrap: 'anywhere' }}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    
                    {msg.type === 'bot' && msg.chart_metadata?.render_chart && (
                        <StockChart metadata={msg.chart_metadata} />
                    )}

                    {msg.latency_ms && (
                      <div className="latency">
                        ✓ Đã tạo trong {Math.round(msg.latency_ms / 1000)}s
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="message bot">
                  <Sparkles className="bot-icon" />
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <div className="dot"></div>
                      <div className="dot"></div>
                      <div className="dot"></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="input-area">
          <div style={{ width: '100%', maxWidth: '800px' }}>
            
            {/* --- THANH QUẢN LÝ THỜI GIAN & KHUNG --- */}
            <div style={{ display: 'flex', gap: '10px', marginBottom: '12px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: '12px', padding: '6px 12px' }}>
                <Calendar size={16} color="var(--text-secondary)" />
                <input 
                  type="date" 
                  value={startDate}
                  onChange={e => setStartDate(e.target.value)}
                  style={{ background: 'transparent', color: 'var(--text-primary)', border: 'none', fontSize: '14px', outline: 'none' }}
                />
              </div>
              <span style={{ color: 'var(--text-secondary)', alignSelf: 'center' }}>đến</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: '12px', padding: '6px 12px' }}>
                <Calendar size={16} color="var(--text-secondary)" />
                <input 
                  type="date" 
                  value={endDate}
                  onChange={e => setEndDate(e.target.value)}
                  style={{ background: 'transparent', color: 'var(--text-primary)', border: 'none', fontSize: '14px', outline: 'none' }}
                />
              </div>
              <select 
                value={interval}
                onChange={e => setInterval(e.target.value)}
                style={{ background: 'var(--bg-input)', color: 'var(--accent)', border: '1px solid var(--accent)', borderRadius: '12px', padding: '6px 16px', fontSize: '14px', fontWeight: 600, outline: 'none', cursor: 'pointer' }}
              >
                <option value="1D">Nến Ngày (1D)</option>
                <option value="1W">Nến Tuần (1W)</option>
                <option value="1M">Nến Tháng (1M)</option>
              </select>
            </div>

            <div className="input-container">
              <button className="action-btn">
                <Paperclip size={20} />
              </button>
              <textarea 
                className="text-input"
                placeholder="Nhập câu hỏi tại đây (VD: Phân tích kỹ thuật HPG đợt này)..."
                value={input}
                onChange={(e) => {
                  setInput(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
                onKeyDown={handleKeyDown}
                rows={1}
              />
              <div className="input-actions">
                <button className="action-btn">
                  <Mic size={20} />
                </button>
                <button 
                  className={`action-btn send-btn ${input.trim() ? 'active' : ''}`}
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                >
                  <Send size={20} />
                </button>
              </div>
            </div>
          </div>
          
          <div className="disclaimer">
            FinBot có thể hiển thị thông tin không chính xác. Hãy cẩn trọng kiểm tra các thông tin quan trọng.
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;