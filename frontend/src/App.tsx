import React, { useState, useEffect, useRef } from 'react';
import { 
  Menu, 
  Plus, 
  MessageSquare, 
  Settings, 
  HelpCircle, 
  Send, 
  Paperclip, 
  Mic, 
  Sparkles, 
  History,
  Compass,
  Lightbulb,
  Code,
  TrendingUp
} from 'lucide-react';
import axios from 'axios';
import './index.css';

// --- Types ---
interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  latency_ms?: number;
}

const API_BASE_URL = 'http://localhost:8080';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [marketOverview, setMarketOverview] = useState<any>(null);
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
        session_id: 'browser-user-' + Date.now()
      });

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response.data.answer,
        timestamp: new Date(),
        latency_ms: response.data.latency_ms
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
          <button className="chat-item">
            <MessageSquare size={16} />
            Phân tích cổ phiếu VNM
          </button>
          <button className="chat-item">
            <MessageSquare size={16} />
            Giá HPG hôm nay thế nào?
          </button>
        </div>

        <div className="sidebar-footer">
          <button className="footer-btn">
            <HelpCircle size={20} />
            Trợ giúp
          </button>
          <button className="footer-btn">
            <History size={20} />
            Hoạt động
          </button>
          <button className="footer-btn">
            <Settings size={20} />
            Cài đặt
          </button>
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
                <button className="suggestion-card" onClick={() => setInput("Viết code Python lấy dữ liệu VNINDEX")}>
                  <div>Code Python<br/>phân tích dữ liệu</div>
                  <div className="card-icon"><Code size={24} /></div>
                </button>
              </div>
            </div>
          ) : (
            <div className="messages-wrapper">
              {messages.map((msg) => (
                <div key={msg.id} className={`message ${msg.type}`}>
                  {msg.type === 'bot' && <Sparkles className="bot-icon" />}
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <div className="message-content">{msg.content}</div>
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
          <div className="input-container">
            <button className="action-btn">
              <Paperclip size={20} />
            </button>
            <textarea 
              className="text-input"
              placeholder="Nhập câu hỏi tại đây..."
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
          <div className="disclaimer">
            FinBot có thể hiển thị thông tin không chính xác. Hãy cẩn trọng kiểm tra các thông tin quan trọng. <a href="#">Quyền riêng tư</a>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;