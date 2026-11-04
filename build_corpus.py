import re, json
def clean(s):
    for a,b in [('\u00ad\n',''),('\u00ad',''),('ﬁ','fi'),('ﬂ','fl'),('ﬀ','ff'),('ﬃ','ffi'),('’',"'"),('‘',"'"),('“','"'),('”','"')]:
        s=s.replace(a,b)
    s=re.sub(r'(?<=[a-z])-\n(?=[a-z])','',s)
    s=re.sub(r'\s*\n\s*',' ',s); s=re.sub(r'\s+',' ',s)
    return s.strip()
meta={
1:("Artificial intelligence literacy in higher and adult education: A scoping literature review","Matthias Carl Laupichler; Alexandra Aster; Jana Schirch; Tobias Raupach",2022,"Since artificial intelligence (AI) is finding","are derived from the results."),
2:("Systematic literature review on opportunities, challenges, and future research recommendations of artificial intelligence in education","Thomas K.F. Chiu; Qi Xia; Xinyan Zhou; Ching Sing Chai; Miaoting Cheng",2023,"Applications of artificial intelligence in education (AIEd) are emerging","future directions of AIEd research."),
3:("Myths, mis- and preconceptions of artificial intelligence: A review of the literature","Arne Bewersdorff; Xiaoming Zhai; Jessica Roberts; Claudia Nerdel",2023,"Artificial Intelligence (AI) is prevalent in nearly","in AI education."),
4:("Opportunities of artificial intelligence for supporting complex problem-solving: Findings from a scoping review","Srecko Joksimovic; Dirk Ifenthaler; Rebecca Marrone; Maarten De Laat; George Siemens",2023,"The research objective of this paper","are being discussed."),
5:("Application and theory gaps during the rise of Artificial Intelligence in Education","Xieling Chen; Haoran Xie; Di Zou; Gwo-Jen Hwang",2020,"Considering the increasing importance","with educational theories."),
6:("A multi-perspective study on Artificial Intelligence in Education: grants, conferences, journals, software tools, institutions, and researchers","Xieling Chen; Haoran Xie; Gwo-Jen Hwang",2020,"With the rapid development of artificial intelligence","Computers & Education: Artificial Intelligence."),
7:("Human-centred learning analytics and AI in education: A systematic literature review","Riordan Alfredo; Vanessa Echeverria; Yueqiao Jin; Lixiang Yan; Zachari Swiecki; Dragan Gasevic; Roberto Martinez-Maldonado",2024,"The rapid expansion of Learning Analytics","human-centred LA/AIED systems."),
8:("A systematic review of artificial intelligence techniques for collaborative learning over the past two decades","Seng Chee Tan; Alwyn Vwen Yen Lee; Min Lee",2022,"This systematic review focuses on publications","suggest future areas of study."),
9:("AI technologies for education: Recent research & future directions","Ke Zhang; Ayse Begum Aslan",2021,"From unique educational perspectives","development efforts."),
10:("Educational applications of artificial intelligence in simulation-based learning: A systematic mapping review","Chih-Pu Dai; Fengfeng Ke",2022,"The field of education has experienced","using simulation-based learning."),
11:("A scoping review on how generative artificial intelligence transforms assessment in higher education","Qi Xia; Xiaojing Weng; Fan Ouyang; Tzung Jin Lin; Thomas K.F. Chiu",2024,"Generative artificial intelligence provides both","teaching."),
12:("A meta systematic review of artificial intelligence in higher education: a call for increased ethics, collaboration, and rigour","Melissa Bond; Hassan Khosravi; Maarten De Laat; Nina Bergdahl; Violeta Negrea; Emily Oxley; Phuong Pham; Sin Wang Chong; George Siemens",2024,"Although the field of Artificial Intelligence in Education","primary and secondary research."),
13:("Artificial intelligence for education and its emphasis on assessment and adversity quotient: a review","Jyoti Mudkanna Gavhane; Reena Pagare",2024,"Purpose - The purpose of this study","evaluation of AQ."),
14:("A comprehensive exploration of artificial intelligence competence frameworks for educators: A critical review","Tamar Mikeladze; Paulien C. Meijer; Roald P. Verhoeff",2024,None,None),
15:("A Systematic Review of Generative AI for Teaching and Learning Practice","Bayode Ogunleye; Kudirat Ibilola Zakariyyah; Oluwaseun Ajao; Olakunle Olayinka; Hemlata Sharma",2024,"The use of generative artificial intelligence (GenAI)","policies for GenAI usage."),
16:("Generative Artificial Intelligence in Tertiary Education: Assessment Redesign Principles and Considerations","Che Yee Lye; Lyndon Lim",2024,"The emergence of generative artificial intelligence","address the fundamental issues in assessments."),
17:("A Systematic Review of AI-Driven Educational Assessment in STEM Education","Fan Ouyang; Tuan Anh Dinh; Weiqi Xu",2023,"Artificial intelligence (AI), as an emerging technology","STEM education."),
18:("Transforming Education: A Comprehensive Review of Generative Artificial Intelligence in Educational Settings through Bibliometric and Content Analysis","Zied Bahroun; Chiraz Anane; Vian Ahmed; Andrew Zacca",2023,"In the ever-evolving era of technological","educators, and policymakers interested in the intersection of GAI and education."),
19:("What Is the Impact of ChatGPT on Education? A Rapid Review of the Literature","Chung Kwan Lo",2023,"An artificial intelligence-based chatbot","on the educational environment."),
20:("Systematic review of research on artificial intelligence applications in higher education - where are the educators?","Olaf Zawacki-Richter; Victoria I. Marin; Melissa Bond; Franziska Gouverneur",2019,"According to various international reports","approaches in the application of AIEd in higher education."),
}
A14=("Recent literature underscores the need for teachers to develop AI competencies with a recognition of the current lack of well-defined competence frameworks. This critical review investigates teachers' Artificial Intelligence (AI) competence frameworks (AI CFTs), analysing their strengths, weaknesses and practical applications for researchers, educators and policymakers. It identifies five distinct types of AI CFTs, each addressing the complexities of AI in its early stages. Notably, frameworks derived from empirical data offer detailed descriptions of competencies, while those based on conceptual models provide broader overviews. Highlighting the need for further empirical research, this review helps identify and understand existing approaches to teacher AI competence development and paves the way for integrating AI CFTs into teacher education, ultimately enhancing educators' preparedness to harness AI in their teaching practices.")
docs=[]
for i in range(1,21):
    title,auth,year,s,e=meta[i]
    if i==14: ab=A14
    else:
        t=clean(open(f'p1/{i}.txt',encoding='utf8').read())
        s2=s.replace(' - ',' – ')
        a=t.find(s); 
        if a<0: a=t.find(s2)
        assert a>=0,(i,'start')
        b=t.find(e,a); assert b>=0,(i,'end')
        ab=t[a:b+len(e)]
        # remove stray keyword block (papers 4-10 where Keywords precede abstract are before 'a'); remove inline 'Keywords: ...' if inside
        ab=ab.replace('AIdriven','AI-driven').replace('Purpose – ','Purpose: ').replace('Purpose - ','Purpose: ')
    docs.append(dict(id=i,title=title,authors=auth,year=year,abstract=ab,words=len(ab.split())))
    print(i,len(ab.split()),'|',ab[:90],'...',ab[-60:])
json.dump(docs,open('app/data/corpus.json','w'),ensure_ascii=False,indent=1)
